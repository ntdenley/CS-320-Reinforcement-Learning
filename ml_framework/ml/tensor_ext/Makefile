main: main.o storage.o 
	cc -o main main.o storage.o

main.o: main.c
	cc -c main.c
storage.o: storage.c storage.h
	cc -c storage.c

run: main
	./main
clean:
	rm main *.o