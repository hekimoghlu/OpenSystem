/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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
#include <ctype.h>

/* Utility library. */

#include <msg.h>
#include <vstream.h>
#include <vstring.h>
#include <compat_va_copy.h>
#include <netstring.h>

/* Application-specific. */

#define STR(x)	vstring_str(x)
#define LEN(x)	VSTRING_LEN(x)

/* netstring_setup - initialize netstring stream */

void    netstring_setup(VSTREAM *stream, int timeout)
{
    vstream_control(stream,
		    CA_VSTREAM_CTL_TIMEOUT(timeout),
		    CA_VSTREAM_CTL_EXCEPT,
		    CA_VSTREAM_CTL_END);
}

/* netstring_except - process netstring stream exception */

void    netstring_except(VSTREAM *stream, int exception)
{
    vstream_longjmp(stream, exception);
}

/* netstring_get_length - read netstring length + terminator */

ssize_t netstring_get_length(VSTREAM *stream)
{
    const char *myname = "netstring_get_length";
    ssize_t len = 0;
    int     ch;
    int     digit;

    for (;;) {
	switch (ch = VSTREAM_GETC(stream)) {
	case VSTREAM_EOF:
	    netstring_except(stream, vstream_ftimeout(stream) ?
			     NETSTRING_ERR_TIME : NETSTRING_ERR_EOF);
	case ':':
	    if (msg_verbose > 1)
		msg_info("%s: read netstring length %ld", myname, (long) len);
	    return (len);
	default:
	    if (!ISDIGIT(ch))
		netstring_except(stream, NETSTRING_ERR_FORMAT);
	    digit = ch - '0';
	    if (len > SSIZE_T_MAX / 10
		|| (len *= 10) > SSIZE_T_MAX - digit)
		netstring_except(stream, NETSTRING_ERR_SIZE);
	    len += digit;
	    break;
	}
    }
}

/* netstring_get_data - read netstring payload + terminator */

VSTRING *netstring_get_data(VSTREAM *stream, VSTRING *buf, ssize_t len)
{
    const char *myname = "netstring_get_data";

    /*
     * Allocate buffer space.
     */
    VSTRING_RESET(buf);
    VSTRING_SPACE(buf, len);

    /*
     * Read the payload and absorb the terminator.
     */
    if (vstream_fread(stream, STR(buf), len) != len)
	netstring_except(stream, vstream_ftimeout(stream) ?
			 NETSTRING_ERR_TIME : NETSTRING_ERR_EOF);
    if (msg_verbose > 1)
	msg_info("%s: read netstring data %.*s",
		 myname, (int) (len < 30 ? len : 30), STR(buf));
    netstring_get_terminator(stream);

    /*
     * Position the buffer.
     */
    VSTRING_AT_OFFSET(buf, len);
    return (buf);
}

/* netstring_get_terminator - absorb netstring terminator */

void    netstring_get_terminator(VSTREAM *stream)
{
    if (VSTREAM_GETC(stream) != ',')
	netstring_except(stream, NETSTRING_ERR_FORMAT);
}

/* netstring_get - read string from netstring stream */

VSTRING *netstring_get(VSTREAM *stream, VSTRING *buf, ssize_t limit)
{
    ssize_t len;

    len = netstring_get_length(stream);
    if (limit && len > limit)
	netstring_except(stream, NETSTRING_ERR_SIZE);
    netstring_get_data(stream, buf, len);
    return (buf);
}

/* netstring_put - send string as netstring */

void    netstring_put(VSTREAM *stream, const char *data, ssize_t len)
{
    const char *myname = "netstring_put";

    if (msg_verbose > 1)
	msg_info("%s: write netstring len %ld data %.*s",
		 myname, (long) len, (int) (len < 30 ? len : 30), data);
    vstream_fprintf(stream, "%ld:", (long) len);
    vstream_fwrite(stream, data, len);
    VSTREAM_PUTC(',', stream);
}

/* netstring_put_multi - send multiple strings as one netstring */

void    netstring_put_multi(VSTREAM *stream,...)
{
    const char *myname = "netstring_put_multi";
    ssize_t total;
    char   *data;
    ssize_t data_len;
    va_list ap;
    va_list ap2;

    /*
     * Initialize argument lists.
     */
    va_start(ap, stream);
    VA_COPY(ap2, ap);

    /*
     * Figure out the total result size.
     */
    for (total = 0; (data = va_arg(ap, char *)) != 0; total += data_len)
	if ((data_len = va_arg(ap, ssize_t)) < 0)
	    msg_panic("%s: bad data length %ld", myname, (long) data_len);
    va_end(ap);
    if (total < 0)
	msg_panic("%s: bad total length %ld", myname, (long) total);
    if (msg_verbose > 1)
	msg_info("%s: write total length %ld", myname, (long) total);

    /*
     * Send the length, content and terminator.
     */
    vstream_fprintf(stream, "%ld:", (long) total);
    while ((data = va_arg(ap2, char *)) != 0) {
	data_len = va_arg(ap2, ssize_t);
	if (msg_verbose > 1)
	    msg_info("%s: write netstring len %ld data %.*s",
		     myname, (long) data_len,
		     (int) (data_len < 30 ? data_len : 30), data);
	if (vstream_fwrite(stream, data, data_len) != data_len)
	    netstring_except(stream, vstream_ftimeout(stream) ?
			     NETSTRING_ERR_TIME : NETSTRING_ERR_EOF);
    }
    va_end(ap2);
    vstream_fwrite(stream, ",", 1);
}

/* netstring_fflush - flush netstring stream */

void    netstring_fflush(VSTREAM *stream)
{
    if (vstream_fflush(stream) == VSTREAM_EOF)
	netstring_except(stream, vstream_ftimeout(stream) ?
			 NETSTRING_ERR_TIME : NETSTRING_ERR_EOF);
}

/* netstring_memcpy - copy data as in-memory netstring */

VSTRING *netstring_memcpy(VSTRING *buf, const char *src, ssize_t len)
{
    vstring_sprintf(buf, "%ld:", (long) len);
    vstring_memcat(buf, src, len);
    VSTRING_ADDCH(buf, ',');
    return (buf);
}

/* netstring_memcat - append data as in-memory netstring */

VSTRING *netstring_memcat(VSTRING *buf, const char *src, ssize_t len)
{
    vstring_sprintf_append(buf, "%ld:", (long) len);
    vstring_memcat(buf, src, len);
    VSTRING_ADDCH(buf, ',');
    return (buf);
}

/* netstring_strerror - convert error number to string */

const char *netstring_strerror(int err)
{
    switch (err) {
	case NETSTRING_ERR_EOF:
	return ("unexpected disconnect");
    case NETSTRING_ERR_TIME:
	return ("time limit exceeded");
    case NETSTRING_ERR_FORMAT:
	return ("input format error");
    case NETSTRING_ERR_SIZE:
	return ("input exceeds size limit");
    default:
	return ("unknown netstring error");
    }
}

 /*
  * Proof-of-concept netstring encoder/decoder.
  * 
  * Usage: netstring command...
  * 
  * Run the command as a child process. Then, convert between plain strings on
  * our own stdin/stdout, and netstrings on the child program's stdin/stdout.
  * 
  * Example (socketmap test server): netstring nc -l 9999
  */
#ifdef TEST
#include <unistd.h>
#include <stdlib.h>
#include <events.h>

static VSTRING *stdin_read_buf;		/* stdin line buffer */
static VSTRING *child_read_buf;		/* child read buffer */
static VSTREAM *child_stream;		/* child stream (full-duplex) */

/* stdin_read_event - line-oriented event handler */

static void stdin_read_event(int event, void *context)
{
    int     ch;

    /*
     * Send a netstring to the child when we have accumulated an entire line
     * of input.
     * 
     * Note: the first VSTREAM_GETCHAR() call implicitly fills the VSTREAM
     * buffer. We must drain the entire VSTREAM buffer before requesting the
     * next read(2) event.
     */
    do {
	ch = VSTREAM_GETCHAR();
	switch (ch) {
	default:
	    VSTRING_ADDCH(stdin_read_buf, ch);
	    break;
	case '\n':
	    NETSTRING_PUT_BUF(child_stream, stdin_read_buf);
	    vstream_fflush(child_stream);
	    VSTRING_RESET(stdin_read_buf);
	    break;
	case VSTREAM_EOF:
	    /* Better: wait for child to terminate. */
	    sleep(1);
	    exit(0);
	}
    } while (vstream_peek(VSTREAM_IN) > 0);
}

/* child_read_event - netstring-oriented event handler */

static void child_read_event(int event, void *context)
{

    /*
     * Read an entire netstring from the child and send the result to stdout.
     * 
     * This is a simplistic implementation that assumes a server will not
     * trickle its data.
     * 
     * Note: the first netstring_get() call implicitly fills the VSTREAM buffer.
     * We must drain the entire VSTREAM buffer before requesting the next
     * read(2) event.
     */
    do {
	netstring_get(child_stream, child_read_buf, 10000);
	vstream_fwrite(VSTREAM_OUT, STR(child_read_buf), LEN(child_read_buf));
	VSTREAM_PUTC('\n', VSTREAM_OUT);
	vstream_fflush(VSTREAM_OUT);
    } while (vstream_peek(child_stream) > 0);
}

int     main(int argc, char **argv)
{
    int     err;

    /*
     * Sanity check.
     */
    if (argv[1] == 0)
	msg_fatal("usage: %s command...", argv[0]);

    /*
     * Run the specified command as a child process with stdin and stdout
     * connected to us.
     */
    child_stream = vstream_popen(O_RDWR, CA_VSTREAM_POPEN_ARGV(argv + 1),
				 CA_VSTREAM_POPEN_END);
    vstream_control(child_stream, CA_VSTREAM_CTL_DOUBLE, CA_VSTREAM_CTL_END);
    netstring_setup(child_stream, 10);

    /*
     * Buffer plumbing.
     */
    stdin_read_buf = vstring_alloc(100);
    child_read_buf = vstring_alloc(100);

    /*
     * Monitor both the child's stdout stream and our own stdin stream. If
     * there is activity on the child stdout stream, read an entire netstring
     * or EOF. If there is activity on stdin, send a netstring to the child
     * when we have read an entire line, or terminate in case of EOF.
     */
    event_enable_read(vstream_fileno(VSTREAM_IN), stdin_read_event, (void *) 0);
    event_enable_read(vstream_fileno(child_stream), child_read_event,
		      (void *) 0);

    if ((err = vstream_setjmp(child_stream)) == 0) {
	for (;;)
	    event_loop(-1);
    } else {
	msg_fatal("%s: %s", argv[1], netstring_strerror(err));
    }
}

#endif
