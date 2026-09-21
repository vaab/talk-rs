#include <signal.h>
#include <unistd.h>

static volatile sig_atomic_t running = 1;

static void stop(int signal_number) {
    (void)signal_number;
    running = 0;
}

int main(void) {
    signal(SIGHUP, stop);
    signal(SIGINT, stop);
    signal(SIGTERM, stop);
    while (running) {
        pause();
    }
    return 0;
}
