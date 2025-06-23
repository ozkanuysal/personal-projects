import speedtest
import logging

# Configure logging
logging.basicConfig(filename='network_speed.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def test_network_speed():
    try:
        # Create a Speedtest object
        st = speedtest.Speedtest()
        st.get_best_server()

        # Perform download and upload speed tests
        download_speed = st.download() / 1_000_000  # Convert to Mbps
        upload_speed = st.upload() / 1_000_000  # Convert to Mbps

        # Get ping (latency)
        ping = st.results.ping

        result = {
            "download_speed_mbps": download_speed,
            "upload_speed_mbps": upload_speed,
            "ping_ms": ping
        }

        # Log the results
        logging.info(f"Download Speed: {result['download_speed_mbps']} Mbps")
        logging.info(f"Upload Speed: {result['upload_speed_mbps']} Mbps")
        logging.info(f"Ping: {result['ping_ms']} ms")

        return result
    except Exception as e:
        logging.error(f"Error during speed test: {e}")
        return None

if __name__ == "__main__":
    speeds = test_network_speed()
    if speeds:
        print(f"Download Speed: {speeds['download_speed_mbps']} Mbps")
        print(f"Upload Speed: {speeds['upload_speed_mbps']} Mbps")
        print(f"Ping: {speeds['ping_ms']} ms")
    else:
        print("Failed to perform network speed test. Check logs for more details.")