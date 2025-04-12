all:
	g++ main.cpp -o main \
	-I/opt/homebrew/opt/opencv/include/opencv4 \
	-L/opt/homebrew/opt/opencv/lib \
	-lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_videoio \
	-std=c++11
	./main

# all:
# 	g++ main.cpp -o main \
# 	-I/opt/homebrew/opt/opencv/include/opencv4 \
# 	-L/opt/homebrew/opt/opencv/lib \
# 	-lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -std=c++11
# 	./main

clean:
	rm -f main
