import http.server
import socketserver
import json
import time
import threading

PORT = 80

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            html = """
            <html>
            <head><title>Web Server Lab</title></head>
            <body>
                <h1>Welcome to D7078E Cloud Security Lab</h1>
                <p>Web Server is running!</p>
                <p>Instance: {}</p>
                <p>Time: {}</p>
                <hr>
                <p><a href="/burn">/burn</a> - CPU-intensive endpoint (5s load)</p>
            </body>
            </html>
            """.format(self.server.hostname, time.strftime("%Y-%m-%d %H:%M:%S"))
            self.wfile.write(html.encode())
            self.log_message("INFO", "Served index page")

        elif self.path == '/burn':
            # CPU-intensive operation
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            start = time.time()
            # Compute-heavy loop (approx 5 seconds on t2.micro)
            result = 0
            for i in range(50000000):
                result += i ** 2
            elapsed = time.time() - start

            response = json.dumps({
                "status": "burn_complete",
                "cpu_time_seconds": round(elapsed, 2),
                "iterations": 50000000
            })
            self.wfile.write(response.encode())
            self.log_message("INFO", f"Burn completed in {elapsed:.2f}s")

        elif self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            response = json.dumps({
                "status": "ok",
                "uptime_seconds": time.time() - self.server.start_time,
                "requests_served": self.server.request_count
            })
            self.wfile.write(response.encode())
            self.log_message("INFO", "Served metrics")

        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    def log_message(self, format, *args):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {format % args}")

if __name__ == '__main__':
    handler = CustomHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        httpd.hostname = "web-server-lab"
        httpd.start_time = time.time()
        httpd.request_count = 0
        print(f"Server running on port {PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped")