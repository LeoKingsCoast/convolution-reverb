# Convolution Reverb

This is a simple C program that takes an audio input, and outputs the same
audio with a reverb effect applied. To apply the effect, the program outputs 
the convolution of the input signal with an impulse response of the reverb 
effect. The linear convolution is computed using the overlap-save algorithm 
over the circular convolution of each block of the input with the impulse 
response.

## Dependencies

- [Portaudio](https://www.portaudio.com/)
- [FFTW](https://www.fftw.org/)

## Build and Run

To build and run the application, follow the steps below:

1. Install the dependencies:

    ```bash
    make install_deps
    ```

1. Build the application. Note: Building the application will require root
access. This is necessary to give the binary the `CAP_IPC_LOCK` capability,
required for it to lock memory pages for real-time safety.

    ```bash
    make
    ```

1. Run the application. You should be able to hear the audio of your machine's 
main input with the reverb applied.

    ```bash
    ./conv-rev
    ```
