# The MIT License (MIT)

# Copyright (c) 2015 İlker Kesen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



# Based on https://github.com/ilkerkesen/tornado-websocket-client-example

import asyncio
import rich_click as click
import websockets

class Client(object):
    def __init__(self, url, timeout, port):
        self.url = url
        self.timeout = timeout
        self.port = port
        self.ws = None


    async def connect(self):
        try:
            # Connect to the WebSocket server
            self.ws = await websockets.connect(self.url, open_timeout=self.timeout)
            print(f"Connected to {self.url}")
            await self.run() # Call run after successful connection
        except Exception as e:
            print(f"Connection failed: {e}")
            self.ws = None


    async def run(self):
        if not self.ws:
            print("WebSocket connection not established.")
            return
        try:
            while True:
                # Receive messages from the server
                msg = await self.ws.recv()
                if msg is None: # Connection closed by server
                    print("Connection closed by server.")
                    break
                print(f"Received message: {msg}")
                # Add message handling logic here if needed
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed unexpectedly.")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if self.ws:
                await self.ws.close()
            self.ws = None



@click.command()
@click.argument("port", type=int)  # Accept positional port argument
def main(port):
    url = f"ws://localhost:{port}" # Use f-string
    client = Client(url, 5, port)
    try:
        asyncio.run(client.connect())
    except KeyboardInterrupt:
        print("Client stopped by user.")

if __name__ == "__main__":
    main()
