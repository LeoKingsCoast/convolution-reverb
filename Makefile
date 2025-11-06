BIN := conv-rev
SRC := main.c convolution.c audio.c
LDLIBS := -lfftw3f -lportaudio -lm -lrt -lasound -pthread

SRCDIR := src
BUILDDIR := build

CFLAGS := -g -Wall -Werror

SRCFILES := $(addprefix $(SRCDIR)/, $(SRC))

override CC := $(if $(filter default undefined,$(origin CC)),gcc,$(CC))

all: $(BUILDDIR)/$(BIN)
.PHONY := all

$(BUILDDIR)/$(BIN): $(SRCFILES)
	mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)
	setcap cap_ipc_lock=+ep $@

install_deps: install_fftw install_portaudio
.PHONY := install_deps

uninstall_deps: uninstall_fftw uninstall_portaudio
.PHONY := uninstall_deps

install_fftw:
	mkdir -p lib
	curl "https://www.fftw.org/fftw-3.3.10.tar.gz" | tar xz -C lib
	cd lib/fftw-3.3.10 && ./configure && ${MAKE} && sudo ${MAKE} install
.PHONY := install_fftw

install_portaudio:
	mkdir -p lib
	curl "https://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz" | tar xz -C lib
	cd lib/portaudio && ./configure && ${MAKE}
.PHONY := install_portaudio

uninstall_fftw:
	cd lib/fftw-3.3.10 && sudo ${MAKE} unisntall
	rm -rf lib/fftw-3.3.10
.PHONY := uninstall_fftw

uninstall_portaudio:
	cd lib/portaudio && ${MAKE} unisntall
	rm -rf lib/portaudio
.PHONY := uninstall_portaudio

clean:
	rm -rf $(BUILDDIR)
.PHONY := clean
