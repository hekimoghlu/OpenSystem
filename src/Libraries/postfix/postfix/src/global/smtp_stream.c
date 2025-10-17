/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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
#include <sys/socket.h>
#include <sys/time.h>
#include <setjmp.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <string.h>			/* FD_ZERO() needs bzero() prototype */
#include <errno.h>

/* Utility library. */

#include <vstring.h>
#include <vstream.h>
#include <vstring_vstream.h>
#include <msg.h>
#include <iostuff.h>

/* Application-specific. */

#include "smtp_stream.h"

/* smtp_timeout_reset - reset per-stream error flags, restart deadline timer */

static void smtp_timeout_reset(VSTREAM *stream)
{
    vstream_clearerr(stream);

    /*
     * Important: the time limit feature must not introduce any system calls
     * when the input is already in the buffer, or when the output still fits
     * in the buffer. Such system calls would really hurt when receiving or
     * sending body content one line at a time.
     */
    if (vstream_fstat(stream, VSTREAM_FLAG_DEADLINE))
	vstream_control(stream, CA_VSTREAM_CTL_START_DEADLINE, CA_VSTREAM_CTL_END);
}

/* smtp_longjmp - raise an exception */

static NORETURN smtp_longjmp(VSTREAM *stream, int err, const char *context)
{

    /*
     * If we failed to write, don't bang our head against the wall another
     * time when closing the stream. In the case of SMTP over TLS, poisoning
     * the socket with shutdown() is more robust than purging the VSTREAM
     * buffer or replacing the write function pointer with dummy_write().
     */
    if (msg_verbose)
	msg_info("%s: %s", context, err == SMTP_ERR_TIME ? "timeout" : "EOF");
    if (vstream_wr_error(stream))
	/* Don't report ECONNRESET (hangup), EINVAL (already shut down), etc. */
	(void) shutdown(vstream_fileno(stream), SHUT_WR);
    vstream_longjmp(stream, err);
}

/* smtp_stream_setup - configure timeout trap */

void    smtp_stream_setup(VSTREAM *stream, int maxtime, int enable_deadline)
{
    const char *myname = "smtp_stream_setup";

    if (msg_verbose)
	msg_info("%s: maxtime=%d enable_deadline=%d",
		 myname, maxtime, enable_deadline);

    vstream_control(stream,
		    CA_VSTREAM_CTL_DOUBLE,
		    CA_VSTREAM_CTL_TIMEOUT(maxtime),
		    enable_deadline ? CA_VSTREAM_CTL_START_DEADLINE
		    : CA_VSTREAM_CTL_STOP_DEADLINE,
		    CA_VSTREAM_CTL_EXCEPT,
		    CA_VSTREAM_CTL_END);
}

/* smtp_flush - flush stream */

void    smtp_flush(VSTREAM *stream)
{
    int     err;

    /*
     * Do the I/O, protected against timeout.
     */
    smtp_timeout_reset(stream);
    err = vstream_fflush(stream);

    /*
     * See if there was a problem.
     */
    if (vstream_ftimeout(stream))
	smtp_longjmp(stream, SMTP_ERR_TIME, "smtp_flush");
    if (err != 0)
	smtp_longjmp(stream, SMTP_ERR_EOF, "smtp_flush");
}

/* smtp_vprintf - write one line to SMTP peer */

void    smtp_vprintf(VSTREAM *stream, const char *fmt, va_list ap)
{
    int     err;

    /*
     * Do the I/O, protected against timeout.
     */
    smtp_timeout_reset(stream);
    vstream_vfprintf(stream, fmt, ap);
    vstream_fputs("\r\n", stream);
    err = vstream_ferror(stream);

    /*
     * See if there was a problem.
     */
    if (vstream_ftimeout(stream))
	smtp_longjmp(stream, SMTP_ERR_TIME, "smtp_vprintf");
    if (err != 0)
	smtp_longjmp(stream, SMTP_ERR_EOF, "smtp_vprintf");
}

/* smtp_printf - write one line to SMTP peer */

void    smtp_printf(VSTREAM *stream, const char *fmt,...)
{
    va_list ap;

    va_start(ap, fmt);
    smtp_vprintf(stream, fmt, ap);
    va_end(ap);
}

/* smtp_fgetc - read one character from SMTP peer */

int     smtp_fgetc(VSTREAM *stream)
{
    int     ch;

    /*
     * Do the I/O, protected against timeout.
     */
    smtp_timeout_reset(stream);
    ch = VSTREAM_GETC(stream);

    /*
     * See if there was a problem.
     */
    if (vstream_ftimeout(stream))
	smtp_longjmp(stream, SMTP_ERR_TIME, "smtp_fgetc");
    if (vstream_feof(stream) || vstream_ferror(stream))
	smtp_longjmp(stream, SMTP_ERR_EOF, "smtp_fgetc");
    return (ch);
}

/* smtp_get - read one line from SMTP peer */

int     smtp_get(VSTRING *vp, VSTREAM *stream, ssize_t bound, int flags)
{
    int     last_char;
    int     next_char;

    /*
     * It's painful to do I/O with records that may span multiple buffers.
     * Allow for partial long lines (we will read the remainder later) and
     * allow for lines ending in bare LF. The idea is to be liberal in what
     * we accept, strict in what we send.
     * 
     * XXX 2821: Section 4.1.1.4 says that an SMTP server must not recognize
     * bare LF as record terminator.
     */
    smtp_timeout_reset(stream);
    last_char = (bound == 0 ? vstring_get(vp, stream) :
		 vstring_get_bound(vp, stream, bound));

    switch (last_char) {

	/*
	 * Do some repair in the rare case that we stopped reading in the
	 * middle of the CRLF record terminator.
	 */
    case '\r':
	if ((next_char = VSTREAM_GETC(stream)) == '\n') {
	    VSTRING_ADDCH(vp, '\n');
	    last_char = '\n';
	    /* FALLTRHOUGH */
	} else {
	    if (next_char != VSTREAM_EOF)
		vstream_ungetc(stream, next_char);
	    break;
	}

	/*
	 * Strip off the record terminator: either CRLF or just bare LF.
	 * 
	 * XXX RFC 2821 disallows sending bare CR everywhere. We remove bare CR
	 * if received before CRLF, and leave it alone otherwise.
	 */
    case '\n':
	vstring_truncate(vp, VSTRING_LEN(vp) - 1);
	while (VSTRING_LEN(vp) > 0 && vstring_end(vp)[-1] == '\r')
	    vstring_truncate(vp, VSTRING_LEN(vp) - 1);
	VSTRING_TERMINATE(vp);
	/* FALLTRHOUGH */

	/*
	 * Partial line: just read the remainder later. If we ran into EOF,
	 * the next test will deal with it.
	 */
    default:
	break;
    }

    /*
     * Optionally, skip over excess input, protected by the same time limit.
     */
    if (last_char != '\n' && (flags & SMTP_GET_FLAG_SKIP)
	&& vstream_feof(stream) == 0 && vstream_ferror(stream) == 0)
	while ((next_char = VSTREAM_GETC(stream)) != VSTREAM_EOF
	       && next_char != '\n')
	     /* void */ ;

    /*
     * EOF is bad, whether or not it happens in the middle of a record. Don't
     * allow data that was truncated because of EOF.
     */
    if (vstream_ftimeout(stream))
	smtp_longjmp(stream, SMTP_ERR_TIME, "smtp_get");
    if (vstream_feof(stream) || vstream_ferror(stream))
	smtp_longjmp(stream, SMTP_ERR_EOF, "smtp_get");
    return (last_char);
}

/* smtp_fputs - write one line to SMTP peer */

void    smtp_fputs(const char *cp, ssize_t todo, VSTREAM *stream)
{
    ssize_t err;

    if (todo < 0)
	msg_panic("smtp_fputs: negative todo %ld", (long) todo);

    /*
     * Do the I/O, protected against timeout.
     */
    smtp_timeout_reset(stream);
    err = (vstream_fwrite(stream, cp, todo) != todo
	   || vstream_fputs("\r\n", stream) == VSTREAM_EOF);

    /*
     * See if there was a problem.
     */
    if (vstream_ftimeout(stream))
	smtp_longjmp(stream, SMTP_ERR_TIME, "smtp_fputs");
    if (err != 0)
	smtp_longjmp(stream, SMTP_ERR_EOF, "smtp_fputs");
}

/* smtp_fwrite - write one string to SMTP peer */

void    smtp_fwrite(const char *cp, ssize_t todo, VSTREAM *stream)
{
    ssize_t err;

    if (todo < 0)
	msg_panic("smtp_fwrite: negative todo %ld", (long) todo);

    /*
     * Do the I/O, protected against timeout.
     */
    smtp_timeout_reset(stream);
    err = (vstream_fwrite(stream, cp, todo) != todo);

    /*
     * See if there was a problem.
     */
    if (vstream_ftimeout(stream))
	smtp_longjmp(stream, SMTP_ERR_TIME, "smtp_fwrite");
    if (err != 0)
	smtp_longjmp(stream, SMTP_ERR_EOF, "smtp_fwrite");
}

/* smtp_fputc - write to SMTP peer */

void    smtp_fputc(int ch, VSTREAM *stream)
{
    int     stat;

    /*
     * Do the I/O, protected against timeout.
     */
    smtp_timeout_reset(stream);
    stat = VSTREAM_PUTC(ch, stream);

    /*
     * See if there was a problem.
     */
    if (vstream_ftimeout(stream))
	smtp_longjmp(stream, SMTP_ERR_TIME, "smtp_fputc");
    if (stat == VSTREAM_EOF)
	smtp_longjmp(stream, SMTP_ERR_EOF, "smtp_fputc");
}
