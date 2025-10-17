/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include <config.h>

#include <stdlib.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#if 0 /* Where were those needed? /confused */
#ifdef HAVE_SYS_PROC_H
#include <sys/proc.h>
#endif

#ifdef HAVE_SYS_TTY_H
#include <sys/tty.h>
#endif
#endif

#ifdef HAVE_TERMIOS_H
#include <termios.h>
#endif

#include "roken.h"

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
rk_get_window_size(int fd, int *lines, int *columns)
{
    char *s;

#if defined(TIOCGWINSZ)
    {
	struct winsize ws;
        int ret;
	ret = ioctl(fd, TIOCGWINSZ, &ws);
	if (ret != -1) {
	    if (lines)
		*lines = ws.ws_row;
	    if (columns)
		*columns = ws.ws_col;
	    return 0;
	}
    }
#elif defined(TIOCGSIZE)
    {
	struct ttysize ts;
        int ret;
	ret = ioctl(fd, TIOCGSIZE, &ts);
	if (ret != -1) {
	    if (lines)
		*lines = ts.ws_lines;
	    if (columns)
		*columns = ts.ts_cols;
	    return 0;
 	}
    }
#elif defined(HAVE__SCRSIZE)
    {
	int dst[2];

 	_scrsize(dst);
	if (lines)
	    *lines = dst[1];
	if (columns)
	    *columns = dst[0];
	return 0;
    }
#elif defined(_WIN32)
    {
        intptr_t fh = 0;
        CONSOLE_SCREEN_BUFFER_INFO sb_info;

        fh = _get_osfhandle(fd);
        if (fh != (intptr_t) INVALID_HANDLE_VALUE &&
            GetConsoleScreenBufferInfo((HANDLE) fh, &sb_info)) {
            if (lines)
                *lines = 1 + sb_info.srWindow.Bottom - sb_info.srWindow.Top;
            if (columns)
                *columns = 1 + sb_info.srWindow.Right - sb_info.srWindow.Left;

            return 0;
        }
    }
#endif
    if (columns) {
    	if ((s = getenv("COLUMNS")))
	    *columns = atoi(s);
	else
	    return -1;
    }
    if (lines) {
	if ((s = getenv("LINES")))
	    *lines = atoi(s);
	else
	    return -1;
    }
    return 0;
}
