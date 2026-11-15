### Disclaimer: Code in this repo was developed with AI assistance
# So-Fi - Basically wi-fi but using sound waves instead of radio waves
#### Submitted to DurHack 2026

## About 
Why use radio waves for wireless communication when you can use sound instead? Ignoring that the noise is so annoying it got us kicked out of the main hacking space, using sound for data transmission provides a real insight into how devices communicate with each other, as you can physically hear every byte of data exchanged.

We implemented **So-Fi**: real time data transmission between multiple devices through a router node, all using sound signals. Assign a unique identifier to each end node device. Transfer data by setting the destination ID, and writing some text or drawing/uploading a 16x16 image to be sent, and hitting send. The sender then initiates a connection with the router, confirms the destination and sends the data one packet at a time. Data is encoded in hexadecimal, with each hex digit assigned its own audio frequency. Error checking of each packet is performed using a 32 bit checksum, and erroneous packets are asked to be resent. If the sender receives no ACK after sending a packet, it is assumed to be lost and is resent. Once the router has received the full transmitted data from the sender, it will forward it to the end node with the target ID, again one packet at a time with the same protocol.

## Running the code
Ensure audio enhancements and/or spatial sound are turned off so your microphone processes the signals. Use `audio_spectrum_test.py` while receiving data to check that your microphone is detecting the signals - the system uses frequencies between 8-12kHz.  

**Running the router**:  
`pip install -r requirements.txt`  
`python router_node.py`  

**Running the endpoint node**:  
`pip install -r requirements.txt`  
`python endpoint_node.py`  

## Potential Future Development
In the future, this project could be expanded to incorporate packet switching between routers on a larger network, and to incorporate a CSMA system so that the entire thing doesn't break if two devices try to communicate at once.  
We could also develop it to use ultrasound, which would be less annoying but also less funny, and would require hardware capable of generating and detecting ultrasound frequencies.