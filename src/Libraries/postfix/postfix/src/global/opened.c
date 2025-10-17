/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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
/* System library. */

#include <sys_defs.h>
#include <stdlib.h>			/* 44BSD stdarg.h uses abort() */
#include <stdarg.h>

/* Utility library. */

#include <msg.h>
#include <vstring.h>

/* Global library. */

#include "opened.h"

/* opened - log that a message was opened */

void    opened(const char *queue_id, const char *sender, long size, int nrcpt,
	               const char *fmt,...)
{
    va_list ap;

    va_start(ap, fmt);
    vopened(queue_id, sender, size, nrcpt, fmt, ap);
    va_end(ap);
}

/* vopened - log that a message was opened */

void    vopened(const char *queue_id, const char *sender, long size, int nrcpt,
		        const char *fmt, va_list ap)
{
    VSTRING *text = vstring_alloc(100);

#define TEXT (vstring_str(text))

    vstring_vsprintf(text, fmt, ap);
    msg_info("%s: from=<%s>, size=%ld, nrcpt=%d%s%s%s",
	     queue_id, sender, size, nrcpt,
	     *TEXT ? " (" : "", TEXT, *TEXT ? ")" : "");
    vstring_free(text);
}
