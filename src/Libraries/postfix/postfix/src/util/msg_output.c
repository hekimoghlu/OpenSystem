/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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
#include <stdarg.h>
#include <errno.h>

/* Utility library. */

#include <mymalloc.h>
#include <vstring.h>
#include <vstream.h>
#include <msg_vstream.h>
#include <stringops.h>
#include <percentm.h>
#include <msg_output.h>

 /*
  * Global scope, to discourage the compiler from doing smart things.
  */
volatile int msg_vprintf_lock;
volatile int msg_text_lock;

 /*
  * Private state.
  */
static MSG_OUTPUT_FN *msg_output_fn = 0;
static int msg_output_fn_count = 0;
static VSTRING *msg_buffer = 0;

/* msg_output - specify output handler */

void    msg_output(MSG_OUTPUT_FN output_fn)
{

    /*
     * Allocate all resources during initialization.
     */
    if (msg_buffer == 0)
	msg_buffer = vstring_alloc(100);

    /*
     * We're not doing this often, so avoid complexity and allocate memory
     * for an exact fit.
     */
    if (msg_output_fn_count == 0)
	msg_output_fn = (MSG_OUTPUT_FN *) mymalloc(sizeof(*msg_output_fn));
    else
	msg_output_fn = (MSG_OUTPUT_FN *) myrealloc((void *) msg_output_fn,
			(msg_output_fn_count + 1) * sizeof(*msg_output_fn));
    msg_output_fn[msg_output_fn_count++] = output_fn;
}

/* msg_printf - format text and log it */

void    msg_printf(int level, const char *format,...)
{
    va_list ap;

    va_start(ap, format);
    msg_vprintf(level, format, ap);
    va_end(ap);
}

/* msg_vprintf - format text and log it */

void    msg_vprintf(int level, const char *format, va_list ap)
{
    int     saved_errno = errno;

    if (msg_vprintf_lock == 0) {
	msg_vprintf_lock = 1;
	/* On-the-fly initialization for debugging test programs only. */
	if (msg_output_fn_count == 0)
	    msg_vstream_init("unknown", VSTREAM_ERR);
	/* OK if terminating signal handler hijacks control before next stmt. */
	vstring_vsprintf(msg_buffer, percentm(format, errno), ap);
	msg_text(level, vstring_str(msg_buffer));
	msg_vprintf_lock = 0;
    }
    errno = saved_errno;
}

/* msg_text - sanitize and log pre-formatted text */

void    msg_text(int level, const char *text)
{
    int     i;

    /*
     * Sanitize the text. Use a private copy if necessary.
     */
    if (msg_text_lock == 0) {
	msg_text_lock = 1;
	/* OK if terminating signal handler hijacks control before next stmt. */
	if (text != vstring_str(msg_buffer))
	    vstring_strcpy(msg_buffer, text);
	printable(vstring_str(msg_buffer), '?');
	/* On-the-fly initialization for debugging test programs only. */
	if (msg_output_fn_count == 0)
	    msg_vstream_init("unknown", VSTREAM_ERR);
	for (i = 0; i < msg_output_fn_count; i++)
	    msg_output_fn[i] (level, vstring_str(msg_buffer));
	msg_text_lock = 0;
    }
}
