/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 5, 2022.
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

#include <stdio.h>
#include <stdarg.h>
#include <errno.h>

#include "sudo_compat.h"
#include "sudo_plugin.h"
#include "sudo_debug.h"
#include "pathnames.h"

static int
sudo_printf_int(int msg_type, const char *fmt, ...)
{
    FILE *fp = stdout;
    FILE *ttyfp = NULL;
    va_list ap;
    int len;

    if (ISSET(msg_type, SUDO_CONV_PREFER_TTY)) {
	/* Try writing to /dev/tty first. */
	ttyfp = fopen(_PATH_TTY, "w");
    }

    switch (msg_type & 0xff) {
    case SUDO_CONV_ERROR_MSG:
	fp = stderr;
	FALLTHROUGH;
    case SUDO_CONV_INFO_MSG:
	va_start(ap, fmt);
	len = vfprintf(ttyfp ? ttyfp : fp, fmt, ap);
	va_end(ap);
	break;
    default:
	len = -1;
	errno = EINVAL;
	break;
    }

    if (ttyfp != NULL)
	fclose(ttyfp);

    return len;
}

sudo_printf_t sudo_printf = sudo_printf_int;
