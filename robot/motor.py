from time import sleep
import RPi.GPIO as gpio #https://pypi.python.org/pypi/RPi.GPIO

class Stepper:
	#instantiate stepper 
	#pins = [stepPin, directionPin, enablePin]
	def __init__(self):
		#setup pins
		self.stepPin = 27
		self.directionPin = 4
		self.enablePin = 17
		self.speed = 1000
		
		#use the broadcom layout for the gpio
		gpio.setmode(gpio.BCM)
		
		#set gpio pins
		gpio.setup(self.stepPin, gpio.OUT)
		gpio.setup(self.directionPin, gpio.OUT)
		gpio.setup(self.enablePin, gpio.OUT)
		
		#set enable to low (i.e. power is NOT going to the motor)
		gpio.output(self.enablePin, False)
		
		print("Stepper initialized (step={0}, direction={1}, enable={2})".format(self.stepPin, self.directionPin, self.enablePin))
	
	#clears GPIO settings
	def cleanGPIO(self):
		gpio.cleanup()

	def pause(self):
		sleep(.2)
		
	
	#step the motor
	# steps = number of steps to take
	# dir = direction stepper will move
	# speed = defines the denominator in the waitTime equation: waitTime = 0.000001/speed. As "speed" is increased, the waitTime between steps is lowered
	# stayOn = defines whether or not stepper should stay "on" or not. If stepper will need to receive a new step command immediately, this should be set to "True." Otherwise, it should remain at "False."
	def step(self, steps, dir, stayOn=False):
		#set enable to high (i.e. power IS going to the motor)
		gpio.output(self.enablePin, True)
		
		#set the output to true for left and false for right
		turnLeft = True
		if (dir == 'clockwise'):
			turnLeft = False;
		elif (dir != 'counter-clockwise'):
			print("STEPPER ERROR: no direction supplied")
			return False
		gpio.output(self.directionPin, turnLeft)

		stepCounter = 0
	
		waitTime = 1.0/self.speed #waitTime controls speed

		while stepCounter < steps*5:
			#gracefully exit if ctr-c is pressed
			#exitHandler.exitPoint(True) #exitHandler.exitPoint(True, cleanGPIO)

			#turning the gpio on and off tells the easy driver to take one step
			gpio.output(self.stepPin, True)
			gpio.output(self.stepPin, False)
			#try:
			#	input("Press enter to continue")
			#except SyntaxError:
			#	pass
			stepCounter += 1
 
			#wait before taking the next step thus controlling rotation speed
			sleep(waitTime)
		
		if (stayOn == False):
			#set enable to low (i.e. power is NOT going to the motor)
			gpio.output(self.enablePin, False)

		#print("stepperDriver complete (turned " + dir + " " + str(steps) + " steps)")

class Platter():
	def __init__(self):
		self.position = 0
		self.PINS = 200
		self.STEPS_PER_REVOLUTION = 400
		self.stepper = Stepper()

	def pin_to_position(self, pin):
		return pin * self.STEPS_PER_REVOLUTION / self.PINS % self.STEPS_PER_REVOLUTION

	def position_to_pin(self, position):
		return int(float(position) / self.STEPS_PER_REVOLUTION * float(self.PINS)) % self.PINS

	def go_to_pin(self, pin):
		starting_pos = self.position
		desired_pos = self.pin_to_position(pin)
		if starting_pos > desired_pos + self.STEPS_PER_REVOLUTION/2:
			desired_pos += self.STEPS_PER_REVOLUTION
		elif desired_pos > starting_pos + self.STEPS_PER_REVOLUTION/2:
			starting_pos += self.STEPS_PER_REVOLUTION

		steps_to_move = desired_pos - starting_pos

		if steps_to_move > 0:
			direction = 'clockwise'
		else:
			direction = 'counter-clockwise'
			steps_to_move *= -1

		print "Moving Pin {5} (postion {2}) ==> Pin {0} (position {1}), by going {6} pins ({3} steps) {4}".format(pin, desired_pos, starting_pos, steps_to_move, direction, self.position_to_pin(starting_pos), self.position_to_pin(steps_to_move))
		self.stepper.step(steps_to_move, direction)
		self.position = desired_pos

	def go_to_pins(self, pins):
		for pin in pins:
			self.go_to_pin(pin)
			self.stepper.pause()

		
		
p = Platter()
p.go_to_pins([100,50,49,51,199,10,199,200])
#p.go_to_pins([10,190,0,199,1,199,0])
p.stepper.cleanGPIO()
