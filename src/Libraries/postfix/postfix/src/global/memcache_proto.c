/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 25, 2024.
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
#include <sys_defs.h>

/* Utility library. */

#include <msg.h>
#include <vstream.h>
#include <vstring.h>
#include <vstring_vstream.h>
#include <compat_va_copy.h>

/* Application-specific. */

#include <memcache_proto.h>

#define STR(x) vstring_str(x)
#define LEN(x) VSTRING_LEN(x)

/* memcache_get - read one line from peer */

int     memcache_get(VSTREAM *stream, VSTRING *vp, ssize_t bound)
{
    int     last_char;
    int     next_char;

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
	    /* FALLTRHOUGH */
	} else {
	    if (next_char != VSTREAM_EOF)
		vstream_ungetc(stream, next_char);

	    /*
	     * Input too long, or EOF
	     */
    default:
	    if (msg_verbose)
		msg_info("%s got %s", VSTREAM_PATH(stream),
			 LEN(vp) < bound ? "EOF" : "input too long");
	    return (-1);
	}

	/*
	 * Strip off the record terminator: either CRLF or just bare LF.
	 */
    case '\n':
	vstring_truncate(vp, VSTRING_LEN(vp) - 1);
	if (VSTRING_LEN(vp) > 0 && vstring_end(vp)[-1] == '\r')
	    vstring_truncate(vp, VSTRING_LEN(vp) - 1);
	VSTRING_TERMINATE(vp);
	if (msg_verbose)
	    msg_info("%s got: %s", VSTREAM_PATH(stream), STR(vp));
	return (0);
    }
}

/* memcache_fwrite - write one blob to peer */

int     memcache_fwrite(VSTREAM *stream, const char *cp, ssize_t todo)
{

    /*
     * Sanity check.
     */
    if (todo < 0)
	msg_panic("memcache_fwrite: negative todo %ld", (long) todo);

    /*
     * Do the I/O.
     */
    if (msg_verbose)
	msg_info("%s write: %.*s", VSTREAM_PATH(stream), (int) todo, cp);
    if (vstream_fwrite(stream, cp, todo) != todo
	|| vstream_fputs("\r\n", stream) == VSTREAM_EOF)
	return (-1);
    else
	return (0);
}

/* memcache_fread - read one blob from peer */

int     memcache_fread(VSTREAM *stream, VSTRING *buf, ssize_t todo)
{

    /*
     * Sanity check.
     */
    if (todo < 0)
	msg_panic("memcache_fread: negative todo %ld", (long) todo);

    /*
     * Do the I/O.
     */
    VSTRING_SPACE(buf, todo);
    VSTRING_AT_OFFSET(buf, todo);
    if (vstream_fread(stream, STR(buf), todo) != todo
	|| VSTREAM_GETC(stream) != '\r'
	|| VSTREAM_GETC(stream) != '\n') {
	if (msg_verbose)
	    msg_info("%s read: error", VSTREAM_PATH(stream));
	return (-1);
    } else {
	vstring_truncate(buf, todo);
	VSTRING_TERMINATE(buf);
	if (msg_verbose)
	    msg_info("%s read: %s", VSTREAM_PATH(stream), STR(buf));
	return (0);
    }
}

/* memcache_vprintf - write one line to peer */

int     memcache_vprintf(VSTREAM *stream, const char *fmt, va_list ap)
{

    /*
     * Do the I/O.
     */
    vstream_vfprintf(stream, fmt, ap);
    vstream_fputs("\r\n", stream);
    if (vstream_ferror(stream))
	return (-1);
    else
	return (0);
}

/* memcache_printf - write one line to peer */

int     memcache_printf(VSTREAM *stream, const char *fmt,...)
{
    va_list ap;
    int     ret;

    va_start(ap, fmt);

    if (msg_verbose) {
	VSTRING *buf = vstring_alloc(100);
	va_list ap2;

	VA_COPY(ap2, ap);
	vstring_vsprintf(buf, fmt, ap2);
	va_end(ap2);
	msg_info("%s write: %s", VSTREAM_PATH(stream), STR(buf));
	vstring_free(buf);
    }

    /*
     * Do the I/O.
     */
    ret = memcache_vprintf(stream, fmt, ap);
    va_end(ap);
    return (ret);
}
