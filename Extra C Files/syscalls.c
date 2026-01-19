#include <sys/stat.h>
#include <errno.h>
#include <stdint.h>
#include "stm32h7xx_hal.h"

//also Project → Properties → C/C++ Build → Settings → MCU GCC Linker → Miscellaneous remove --specs=nosys.specs
//in main.c after HAL_init(): setvbuf(stdout, NULL, _IONBF, 0);, will fix printf() to use uart

int _close(int file)
{
    return -1;
}

int _fstat(int file, struct stat *st)
{
    st->st_mode = S_IFCHR;
    return 0;
}

int _isatty(int file)
{
    return 1;
}

int _lseek(int file, int ptr, int dir)
{
    return 0;
}

int _read(int file, char *ptr, int len)
{
    return 0;
}

/* Redirect printf() to UART2 if you want later */
extern UART_HandleTypeDef huart3;

int _write(int file, char *ptr, int len)
{
    HAL_UART_Transmit(&huart3, (uint8_t*)ptr, len, HAL_MAX_DELAY);
    return len;
}

int _getpid(void)
{
    return 1;
}

int _kill(int pid, int sig)
{
    errno = EINVAL;
    return -1;
}