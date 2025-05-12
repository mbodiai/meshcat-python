"""MeshCat server (asyncio/FastAPI edition)

This file replaces the legacy Tornado implementation with an asyncio-native stack
while **keeping the public symbols and module path unchanged** so that external
imports such as

```python
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
```
continue to work.

Main building blocks
====================
* FastAPI (ASGI) – static file hosting and WebSocket endpoint.
* uvicorn – runs the ASGI application.
* zmq.asyncio – REP socket that bridges Python ↔ browser.
* rich-click – coloured CLI used instead of argparse.
* Pydantic – (optional) message validation can be layered later; for now we
  keep the raw binary protocol to stay 100 % wire-compatible.

The functional behaviour is intentionally identical to the old server so that
unit tests and downstream code (e.g. `meshcat.visualizer.ViewerWindow`) do notw
need any changes.
"""

import atexit
import asyncio
import base64
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
import time

import zmq
import zmq.asyncio
import rich_click as click
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles


from meshcat.servers.tree import SceneTree, TreeNode, walk, find_node

# ────────────────────────────────────────────────────────────────────────────
# Constants & configuration
# ────────────────────────────────────────────────────────────────────────────

VIEWER_ROOT = Path(__file__).parent / ".." / "viewer" / "dist"
VIEWER_HTML = "index.html"

DEFAULT_FILESERVER_PORT = 7005
MAX_ATTEMPTS = 1000
DEFAULT_ZMQ_METHOD = "tcp"
DEFAULT_ZMQ_PORT = 6005
DEFAULT_HOST = "127.0.0.1"
# MeshCat command names
MESHCAT_COMMANDS = [
    "set_transform",
    "set_object",
    "delete",
    "set_property",
    "set_animation",
]

# Extended commands that are passed through untouched – kept for completeness
EXTENDED_MESHCAT_COMMANDS = ["start_recording", "stop_recording"]

# ────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ────────────────────────────────────────────────────────────────────────────


def _capture(pattern: str, s: str) -> str:
    import re

    match = re.match(pattern, s)
    if not match:
        raise ValueError(f"Could not match {s!r} with pattern {pattern!r}")
    return match.groups()[0]


def match_zmq_url(line: str) -> str:  # Public API retained for tests
    return _capture(r"^zmq_url=(.*)$", line)


def match_web_url(line: str) -> str:
    return _capture(r"^web_url=(.*)$", line)


def _find_available_port(func, default_port: int, max_attempts: int = MAX_ATTEMPTS, **kwargs):
    """Call *func* with successive port numbers until it succeeds.

    Returns ``(result, port)`` where *result* is whatever *func* returned.
    """
    for i in range(max_attempts):
        port = default_port + i
        try:
            return func(port, **kwargs), port
        except (OSError, zmq.error.ZMQError):
            print(f"Port {port} in use, trying another…", file=sys.stderr)
    raise RuntimeError(f"Could not find a free port starting at {default_port}")


# ────────────────────────────────────────────────────────────────────────────
# Bridge – the heart of the server
# ────────────────────────────────────────────────────────────────────────────

ctx = zmq.asyncio.Context()


class ZMQWebSocketBridge:
    """Bridges ZMQ REP <--> WebSocket messages."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        zmq_url: str | None = None,
        ws_port: int | None = None,
        certfile: str | None = None,
        keyfile: str | None = None,
        ngrok_http_tunnel: bool = False,
    ):
        self.host: str = host
        self.websocket_pool: set[WebSocket] = set()
        self.tree: TreeNode = SceneTree()

        # ─── ZMQ socket ───────────────────────────────────────────────────
        if zmq_url is None:

            def _bind(port: int):
                return self._setup_zmq(f"{DEFAULT_ZMQ_METHOD}://{self.host}:{port}")

            (self.zmq_socket, self.zmq_url), _ = _find_available_port(_bind, DEFAULT_ZMQ_PORT)
        else:
            self.zmq_socket, self.zmq_url = self._setup_zmq(zmq_url)

        # ─── FastAPI app ───────────────────────────────────────────────────
        if ws_port is None:
            _, self.ws_port = _find_available_port(lambda p: p, DEFAULT_FILESERVER_PORT)
        else:
            self.ws_port = ws_port

        protocol = "http"
        if certfile or keyfile:
            # Users can still serve HTTPS via uvicorn's --ssl-keyfile, but we
            # keep the string for backward-compat with the old server.
            protocol = "https"

        self.web_url = f"{protocol}://{self.host}:{self.ws_port}/static/"

        if ngrok_http_tunnel and protocol == "http":
            try:
                import pyngrok.conf  # type: ignore
                import pyngrok.ngrok  # type: ignore

                config = pyngrok.conf.PyngrokConfig(start_new_session=True)
                url = pyngrok.ngrok.connect(self.ws_port, "http", pyngrok_config=config)
                self.web_url = (url.public_url if not isinstance(url, str) else url) + "/static/"
                atexit.register(pyngrok.ngrok.kill)
            except ImportError as e:
                if "pyngrok" in str(e):
                    raise RuntimeError("pyngrok is required for --ngrok_http_tunnel") from e

        self.app = self._make_app()

    # ─────────────────────────── FastAPI wiring ──────────────────────────

    def _make_app(self) -> FastAPI:
        """Build the FastAPI application, including lifespan hooks."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Start background ZMQ listener
            task = asyncio.create_task(self._zmq_loop())
            try:
                yield
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        app = FastAPI(lifespan=lifespan)
        app.mount(
            "/static",
            StaticFiles(directory=str(VIEWER_ROOT), html=True),
            name="static",
        )
        app.add_api_websocket_route("/", self._ws_endpoint)
        # expose for tests
        app.bridge = self  # type: ignore[attr-defined]
        return app

    # ─────────────────────────── ZMQ side ────────────────────────────────

    def _setup_zmq(self, url: str):
        sock = ctx.socket(zmq.REP)
        sock.bind(url)
        return sock, url

    async def _zmq_loop(self):
        """Receive multipart frames from the Python side and act on them."""
        while True:
            frames: list[bytes] = await self.zmq_socket.recv_multipart()
            await self._handle_zmq(frames)
            
    async def _forward_to_websockets(self, data: bytes):
        """Broadcast *data* to all connected WebSockets."""
        if not self.websocket_pool:
            return
        await asyncio.gather(*(ws.send_bytes(data) for ws in self.websocket_pool), return_exceptions=True)

    async def _handle_zmq(self, frames: list[bytes]):
        cmd = frames[0].decode()

        if cmd == "url":  # client requests the viewer URL
            await self.zmq_socket.send(self.web_url.encode())
            return

        if cmd == "wait":
            # Poll until a WebSocket is connected, then reply "ok"
            if self.websocket_pool:
                await self.zmq_socket.send(b"ok")
            else:
                # try again shortly without blocking the event loop
                await asyncio.sleep(0.1)
                await self._handle_zmq(frames)
            return

        if cmd == "set_target":
            await self._forward_to_websockets(frames[2])
            await self.zmq_socket.send(b"ok")
            return

        if cmd == "capture_image":
            if self.websocket_pool:
                await self._forward_to_websockets(frames[2])
                # defer reply: WebSocket will send bytes via send_image()
                self._pending_capture = True  # type: ignore[attr-defined]
            else:
                # no client yet – retry a bit later
                await asyncio.sleep(0.3)
                await self._handle_zmq(frames)
            return

        # Regular meshcat commands with 3-frame layout
        if cmd in MESHCAT_COMMANDS:
            if len(frames) != 3:
                await self.zmq_socket.send(b"error: expected 3 frames")
                return
            path_str = frames[1].decode()
            data = frames[2]
            path = [p for p in path_str.split("/") if p]

            cache_hit = (
                cmd == "set_object" and find_node(self.tree, path).object and find_node(self.tree, path).object == data
            )
            if not cache_hit and self.websocket_pool:
                await asyncio.gather(*(ws.send_bytes(data) for ws in self.websocket_pool), return_exceptions=True)


            # mutate scene tree — mirrors old logic
            node = find_node(self.tree, path)
            if cmd == "set_transform":
                node.transform = data
            elif cmd == "set_object":
                node.object = data
                node.properties = []
            elif cmd == "set_property":
                node.properties.append(data)
            elif cmd == "set_animation":
                node.animation = data
            elif cmd == "delete":
                if path:
                    parent = find_node(self.tree, path[:-1])
                    child = path[-1]
                    if child in parent:
                        del parent[child]
                else:
                    self.tree = SceneTree()

            await self.zmq_socket.send(b"ok")
            return

        if cmd == "get_scene":
            html = self._generate_static_html()
            await self.zmq_socket.send(html.encode())
            return

        await self.zmq_socket.send(b"error: unrecognized command")

     
    async def _ws_endpoint(self, websocket: WebSocket):
        from mbcore.log import info
        await websocket.accept()
        self.websocket_pool.add(websocket)
        info(f"WebSocket opened: {websocket.client}")

        # send current scene to the newcomer
        await self._send_scene(websocket)

        try:
            while True:
                message = await websocket.receive_text()
                try:
                    payload = json.loads(message)["data"]
                    await self._send_image(payload)
                except Exception as err:  # noqa: BLE001
                    print(err, file=sys.stderr)
        except WebSocketDisconnect:
            self.websocket_pool.discard(websocket)
            info(f"WebSocket closed: {websocket.client}")

    async def _send_image(self, data_url: str):
        """Receive base64 png from the browser and reply over ZMQ."""
        mime, img_code = data_url.split(",", 1)
        img_bytes = base64.b64decode(img_code)
        await self.zmq_socket.send(img_bytes)

    async def _send_scene(self, ws: WebSocket):
        for node in walk(self.tree):
            if node.object is not None:
                await ws.send_bytes(node.object)
            for p in node.properties:
                await ws.send_bytes(p)
            if node.transform is not None:
                await ws.send_bytes(node.transform)
            if node.animation is not None:
                await ws.send_bytes(node.animation)

    def _generate_static_html(self) -> str:
        """Return a self-contained HTML snapshot of the current scene."""
        drawing_commands = ""
        for node in walk(self.tree):
            if node.object is not None:
                drawing_commands += _create_command(node.object)
            for p in node.properties:
                drawing_commands += _create_command(p)
            if node.transform is not None:
                drawing_commands += _create_command(node.transform)
            if node.animation is not None:
                drawing_commands += _create_command(node.animation)

        main_js = (VIEWER_ROOT / "main.min.js").read_text()
        return f"""
<!DOCTYPE html>
<html>
  <head><meta charset=utf-8><title>MeshCat</title></head>
  <body>
    <div id=\"meshcat-pane\"></div>
    <script>{main_js}</script>
    <script>
      var viewer = new MeshCat.Viewer(document.getElementById('meshcat-pane'));
      {drawing_commands}
    </script>
    <style>
      body {{margin:0;}}
      #meshcat-pane {{width:100vw;height:100vh;overflow:hidden;}}
    </style>
    <script id=\"embedded-json\"></script>
  </body>
</html>
"""


# ────────────────────────────────────────────────────────────────────────────
# Helper functions (kept from original)
# ────────────────────────────────────────────────────────────────────────────


def _create_command(data: bytes) -> str:
    """Encode a binary draw command into JS that MeshCat viewer understands."""
    return (
        '\nfetch("data:application/octet-binary;base64,{}")'.format(base64.b64encode(data).decode())
        + "\n  .then(r => r.arrayBuffer())"
        + "\n  .then(buf => viewer.handle_command_bytearray(new Uint8Array(buf)));\n"
    )


# ────────────────────────────────────────────────────────────────────────────
# Public helper – unchanged signature (but now in-process, no subprocess)
# ────────────────────────────────────────────────────────────────────────────


def start_zmq_server_as_subprocess(*, zmq_url: str | None = None, server_args: list[str] | None = None):
    """Start the MeshCat server in a *background thread* and return a tuple
    ``(proc_like, zmq_url, web_url)`` that is API-compatible with the old
    subprocess-based helper.

    * ``proc_like`` is a minimal wrapper exposing ``kill()``, ``wait()`` and
      ``poll()`` so existing cleanup code keeps working.  Internally it talks
      to the running :class:`uvicorn.Server` instance.
    """
    import threading
    import uvicorn  # type: ignore

    # -------------------------------------------------------------
    # Parse *subset* of legacy CLI flags so tests keep working
    # -------------------------------------------------------------
    ngrok_http_tunnel = False
    if server_args:
        if "--ngrok_http_tunnel" in server_args:
            ngrok_http_tunnel = True
        # '--zmq-url' flag should have a value right after it
        if "--zmq-url" in server_args:
            flag_index = server_args.index("--zmq-url")
            try:
                zmq_url = server_args[flag_index + 1]
            except IndexError:  # pragma: no cover
                raise ValueError("--zmq-url flag given without value") from None

    # -------------------------------------------------------------
    # Instantiate the bridge (binds REP socket immediately)
    # -------------------------------------------------------------
    bridge = ZMQWebSocketBridge(
        zmq_url=zmq_url,
        ngrok_http_tunnel=ngrok_http_tunnel,
    )

    # -------------------------------------------------------------
    # Spin up the ASGI server in a daemon thread so this call
    # returns immediately *after* startup.
    # -------------------------------------------------------------
    config = uvicorn.Config(
        bridge.app,
        host=bridge.host,
        port=bridge.ws_port,
        log_level="warning",
    )
    server = uvicorn.Server(config=config)

    def _run():
        # Blocks until server.should_exit is set to True
        asyncio.run(server.serve())

    thread = threading.Thread(target=_run, name="meshcat-uvicorn", daemon=True)
    thread.start()

    # Wait until uvicorn has marked itself as started
    start_time = time.time()
    while not server.started and time.time() - start_time < 10:
        pass  # tiny spin – only runs for a few ms

    # -------------------------------------------------------------
    # Provide a proc-like wrapper so external code can shut it down
    # -------------------------------------------------------------
    class _DummyProc:
        def __init__(self, srv: uvicorn.Server, t: threading.Thread):
            self._srv = srv
            self._thread = t

        def kill(self):
            self._srv.should_exit = True

        def wait(self, timeout: float | None = None):
            self._thread.join(timeout)
            return 0

        def poll(self):
            return None if self._thread.is_alive() else 0

    dummy_proc = _DummyProc(server, thread)

    # Ensure the server is stopped at interpreter shutdown
    atexit.register(dummy_proc.kill)

    return dummy_proc, bridge.zmq_url, bridge.web_url


# ────────────────────────────────────────────────────────────────────────────
# CLI (rich-click) – replaces old argparse main
# ────────────────────────────────────────────────────────────────────────────


@click.command()  # type: ignore[attr-defined]
@click.option("--zmq-url", type=str, default=None, help="Bind ZMQ socket to this full URL (e.g. tcp://127.0.0.1:6001)")
@click.option("--host", type=str, default=DEFAULT_HOST, help="Bind server to this host")
@click.option("--ws-port", type=int, default=DEFAULT_FILESERVER_PORT, help="Bind WebSocket server to this port")
@click.option("--open/--no-open", default=False, help="Automatically open the viewer in a browser tab.")  # type: ignore[attr-defined]
@click.option("--certfile", type=str, default=None, help="SSL certificate file for HTTPS (passed to uvicorn)")  # type: ignore[attr-defined]
@click.option("--keyfile", type=str, default=None, help="SSL key file for HTTPS (passed to uvicorn)")  # type: ignore[attr-defined]
@click.option("--ngrok_http_tunnel", is_flag=True, help="Expose HTTP server via ngrok tunnel (requires pyngrok)")  # type: ignore[attr-defined]
def _cli(zmq_url: str | None, host: str, ws_port: int, open: bool, certfile: str | None, keyfile: str | None, ngrok_http_tunnel: bool):
    """Run the MeshCat bridge inline (used by `python -m meshcat.servers.zmqserver`)."""

    bridge = ZMQWebSocketBridge(
        **{
            **({"zmq_url": zmq_url} if zmq_url else {}),
            **({"certfile": certfile} if certfile else {}),
            **({"keyfile": keyfile} if keyfile else {}),
            **({"ngrok_http_tunnel": ngrok_http_tunnel} if ngrok_http_tunnel else {}),
            **({"host": host} if host else {}),
            **({"ws_port": ws_port} if ws_port else {}),
        }

    )

    # Print URLs for parent process / user – **do not change this format**.
    print(f"zmq_url={bridge.zmq_url}")
    print(f"web_url={bridge.web_url}")

    if open:
        import webbrowser

        webbrowser.open(bridge.web_url, new=2)

    import uvicorn  # type: ignore  # Imported late so that libraries can monkey-patch if needed.

    uvicorn.run(
        bridge.app,
        host=bridge.host,
        port=bridge.ws_port,
        ssl_keyfile=keyfile,
        ssl_certfile=certfile,
        log_level="info",
    )


# When executed as a module:  python -m meshcat.servers.zmqserver
if __name__ == "__main__":
    _cli()
