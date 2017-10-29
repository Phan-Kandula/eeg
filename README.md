# Mouse for paralyzed patients
These scripts can allow one to move the mouse of your computer through your brain activity. The Arduino monitors the active part of your brain at an interval and the user is required to think of one of the following depending on the direction the mouse is needed to move.

Right = Imagine moving your right arm and leg
Left = Imagine moving your left arm and leg
Up = Try solving a hard math problem or anything that intellectually challenges you to stimulate your frontal lobe. Requires concentration.
Down = Imagine navigating through your house.

 Note: You may use another setup and the neural network can identify your arrangement if you the setup remains consistent. The setup above is just an example or a starting point.

This software uses a neural networks and other machine learning algorithms that require training.

Requirements:
Python 3.5
Arduino
Electrodes
Administrator Access to computer

Installation guide:
1. Install python 3.5 from https://www.python.org/downloads/
2. Install the following python packages.
	* numpy
	* sklearn
	* scipy
	* pandas
	* pyserial
	* Keras

	Use "pip install <Package Name>" in your shell to install.

3. Install tensor flow from https://www.tensorflow.org/install/ and follow their guide.
4. Install the Arduino Ide from https://www.arduino.cc/en/Main/Software

Training
1. Setup the Arduino by connecting it to your computer and attaching electrodes to the analog ports. It is advised to use different colored electrodes for differentiation.
2. Upload the Arduino code into your Arduino through the Arduino IDE. Note the port in which your Arduino is connected to your laptop.
3. Run the datahandling.py script and input the port when asked.
4. Input the direction you want to train the neural network.
5. Think of one of the following(The direction you want to train your network) when a red light appears on your Arduino.
	Right = Imagine moving your right arm and leg
	Left = Imagine moving your left arm and leg
	Up = Try solving a hard math in your mindproblem or anything that intellectually challenges you to stimulate your frontal lobe. Requires concentration.
	Down = Imagine navigating through your house.
	
6. You can also perform the thinking process throughout the entire time instead of waiting till the light appears. The arduino only monitors after the light appears.
7.It is advised to spend as much time as possible training for each direction. There will be an improved performance through more training. It is suggested to spend around 15 minutes for each direction.
8. Training is required only once per user or when performance is not to satisfaction.
