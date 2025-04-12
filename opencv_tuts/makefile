filename ?= main

run:
	g++ $(filename).cpp -o $(filename) \
	-I/opt/homebrew/opt/opencv/include/opencv4 \
	-L/opt/homebrew/opt/opencv/lib \
	-lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_videoio \
	-std=c++11
	./$(filename)

clean:
	rm -f $(filename)