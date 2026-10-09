import asyncio
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Server:
    def __init__(self, server_id, capacity):
        self.server_id = server_id
        self.capacity = capacity
        self.active_connections = 0

    async def handle_request(self, request_id):
        self.active_connections += 1

        load_percentage = (self.active_connections / self.capacity) * 100
        logger.info(f"Request {request_id} routed to Server {self.server_id}. Active connections: {self.active_connections}/{self.capacity} ({load_percentage:.1f}%)")

        if load_percentage >= 80.0:
            logger.warning(f"ALERT: Server {self.server_id} is approaching 100% CPU load! Current load: {load_percentage:.1f}%")

        # Simulate processing time
        processing_time = random.uniform(0.1, 1.0)
        await asyncio.sleep(processing_time)

        self.active_connections -= 1
        logger.info(f"Request {request_id} completed on Server {self.server_id}. Active connections: {self.active_connections}/{self.capacity}")

class LoadBalancer:
    def __init__(self, servers):
        self.servers = servers

    def get_server(self):
        # Least connections algorithm
        return min(self.servers, key=lambda s: s.active_connections)

    async def route_request(self, request_id):
        server = self.get_server()
        if server.active_connections >= server.capacity:
            logger.error(f"Request {request_id} rejected. Server {server.server_id} is at full capacity.")
            return

        await server.handle_request(request_id)

async def simulate():
    # Initialize 3 servers with a capacity of 10 each
    servers = [Server(i, capacity=10) for i in range(1, 4)]
    lb = LoadBalancer(servers)

    logger.info("Starting load balancer simulation with 50 concurrent requests...")

    tasks = []
    for i in range(1, 51):
        tasks.append(asyncio.create_task(lb.route_request(i)))

    await asyncio.gather(*tasks)

    logger.info("Simulation completed.")

if __name__ == "__main__":
    asyncio.run(simulate())
