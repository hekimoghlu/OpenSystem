/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 10, 2022.
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
/* System libraries. */

#include <sys_defs.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>

/* Application-specific. */

#include "msg.h"
#include "msg_output.h"

 /*
  * Default is verbose logging off.
  */
int     msg_verbose = 0;

 /*
  * Private state.
  */
static MSG_CLEANUP_FN msg_cleanup_fn = 0;
static int msg_error_count = 0;
static int msg_error_bound = 13;

 /*
  * The msg_exiting flag prevents us from recursively reporting an error with
  * msg_fatal*() or msg_panic(), and provides a first-level safety net for
  * optional cleanup actions against signal handler re-entry problems. Note
  * that msg_vprintf() implements its own guard against re-entry.
  * 
  * XXX We specify global scope, to discourage the compiler from doing smart
  * things.
  */
volatile int msg_exiting = 0;

/* msg_info - report informative message */

void    msg_info(const char *fmt,...)
{
    va_list ap;

    va_start(ap, fmt);
    vmsg_info(fmt, ap);
    va_end(ap);
}

void    vmsg_info(const char *fmt, va_list ap)
{
    msg_vprintf(MSG_INFO, fmt, ap);
}

/* msg_warn - report warning message */

void    msg_warn(const char *fmt,...)
{
    va_list ap;

    va_start(ap, fmt);
    vmsg_warn(fmt, ap);
    va_end(ap);
}

void    vmsg_warn(const char *fmt, va_list ap)
{
    msg_vprintf(MSG_WARN, fmt, ap);
}

/* msg_error - report recoverable error */

void    msg_error(const char *fmt,...)
{
    va_list ap;

    va_start(ap, fmt);
    vmsg_error(fmt, ap);
    va_end(ap);
}

void    vmsg_error(const char *fmt, va_list ap)
{
    msg_vprintf(MSG_ERROR, fmt, ap);
    if (++msg_error_count >= msg_error_bound)
	msg_fatal("too many errors - program terminated");
}

/* msg_fatal - report error and terminate gracefully */

NORETURN msg_fatal(const char *fmt,...)
{
    va_list ap;

    va_start(ap, fmt);
    vmsg_fatal(fmt, ap);
    /* NOTREACHED */
}

NORETURN vmsg_fatal(const char *fmt, va_list ap)
{
    if (msg_exiting++ == 0) {
	msg_vprintf(MSG_FATAL, fmt, ap);
	if (msg_cleanup_fn)
	    msg_cleanup_fn();
    }
    sleep(1);
    /* In case we're running as a signal handler. */
    _exit(1);
}

/* msg_fatal_status - report error and terminate gracefully */

NORETURN msg_fatal_status(int status, const char *fmt,...)
{
    va_list ap;

    va_start(ap, fmt);
    vmsg_fatal_status(status, fmt, ap);
    /* NOTREACHED */
}

NORETURN vmsg_fatal_status(int status, const char *fmt, va_list ap)
{
    if (msg_exiting++ == 0) {
	msg_vprintf(MSG_FATAL, fmt, ap);
	if (msg_cleanup_fn)
	    msg_cleanup_fn();
    }
    sleep(1);
    /* In case we're running as a signal handler. */
    _exit(status);
}

/* msg_panic - report error and dump core */

NORETURN msg_panic(const char *fmt,...)
{
    va_list ap;

    va_start(ap, fmt);
    vmsg_panic(fmt, ap);
    /* NOTREACHED */
}

NORETURN vmsg_panic(const char *fmt, va_list ap)
{
    if (msg_exiting++ == 0) {
	msg_vprintf(MSG_PANIC, fmt, ap);
    }
    sleep(1);
    abort();					/* Die! */
    /* In case we're running as a signal handler. */
    _exit(1);					/* DIE!! */
}

/* msg_cleanup - specify cleanup routine */

MSG_CLEANUP_FN msg_cleanup(MSG_CLEANUP_FN cleanup_fn)
{
    MSG_CLEANUP_FN old_fn = msg_cleanup_fn;

    msg_cleanup_fn = cleanup_fn;
    return (old_fn);
}

/* msg_error_limit - set error message counter limit */

int     msg_error_limit(int limit)
{
    int     old = msg_error_bound;

    msg_error_bound = limit;
    return (old);
}

/* msg_error_clear - reset error message counter */

void    msg_error_clear(void)
{
    msg_error_count = 0;
}
