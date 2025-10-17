/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 9, 2022.
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
/*
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <sys/ioctl.h>
#include <stdlib.h>
#include <unistd.h>
#include <termios.h>		/* for struct winsize on HP-UX */
#include <limits.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_util.h"

static int
get_ttysize_ioctl(int *rowp, int *colp)
{
    struct winsize wsize;
    debug_decl(get_ttysize_ioctl, SUDO_DEBUG_UTIL);

    if (ioctl(STDERR_FILENO, TIOCGWINSZ, &wsize) == 0 &&
	wsize.ws_row != 0 && wsize.ws_col  != 0) {
	*rowp = wsize.ws_row;
	*colp = wsize.ws_col;
	debug_return_int(0);
    }
    debug_return_int(-1);
}

void
sudo_get_ttysize_v1(int *rowp, int *colp)
{
    debug_decl(sudo_get_ttysize, SUDO_DEBUG_UTIL);

    if (get_ttysize_ioctl(rowp, colp) == -1) {
	char *p;

	/* Fall back on $LINES and $COLUMNS. */
	if ((p = getenv("LINES")) == NULL ||
	    (*rowp = sudo_strtonum(p, 1, INT_MAX, NULL)) <= 0) {
	    *rowp = 24;
	}
	if ((p = getenv("COLUMNS")) == NULL ||
	    (*colp = sudo_strtonum(p, 1, INT_MAX, NULL)) <= 0) {
	    *colp = 80;
	}
    }

    debug_return;
}
