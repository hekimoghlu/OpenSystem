/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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

#include "sys_defs.h"
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <vstring.h>
#include <vstream.h>
#include <vstring_vstream.h>
#include <connect.h>
#include <hex_quote.h>
#include <dict.h>
#include <stringops.h>
#include <dict_tcp.h>

/* Application-specific. */

typedef struct {
    DICT    dict;			/* generic members */
    VSTRING *raw_buf;			/* raw I/O buffer */
    VSTRING *hex_buf;			/* quoted I/O buffer */
    VSTREAM *fp;			/* I/O stream */
} DICT_TCP;

#define DICT_TCP_MAXTRY	10		/* attempts before giving up */
#define DICT_TCP_TMOUT	100		/* connect/read/write timeout */
#define DICT_TCP_MAXLEN	4096		/* server reply size limit */

#define STR(x)		vstring_str(x)

/* dict_tcp_connect - connect to TCP server */

static int dict_tcp_connect(DICT_TCP *dict_tcp)
{
    int     fd;

    /*
     * Connect to the server. Enforce a time limit on all operations so that
     * we do not get stuck.
     */
    if ((fd = inet_connect(dict_tcp->dict.name, NON_BLOCKING, DICT_TCP_TMOUT)) < 0) {
	msg_warn("connect to TCP map %s: %m", dict_tcp->dict.name);
	return (-1);
    }
    dict_tcp->fp = vstream_fdopen(fd, O_RDWR);
    vstream_control(dict_tcp->fp,
		    CA_VSTREAM_CTL_TIMEOUT(DICT_TCP_TMOUT),
		    CA_VSTREAM_CTL_END);

    /*
     * Allocate per-map I/O buffers on the fly.
     */
    if (dict_tcp->raw_buf == 0) {
	dict_tcp->raw_buf = vstring_alloc(10);
	dict_tcp->hex_buf = vstring_alloc(10);
    }
    return (0);
}

/* dict_tcp_disconnect - disconnect from TCP server */

static void dict_tcp_disconnect(DICT_TCP *dict_tcp)
{
    (void) vstream_fclose(dict_tcp->fp);
    dict_tcp->fp = 0;
}

/* dict_tcp_lookup - request TCP server */

static const char *dict_tcp_lookup(DICT *dict, const char *key)
{
    DICT_TCP *dict_tcp = (DICT_TCP *) dict;
    const char *myname = "dict_tcp_lookup";
    int     tries;
    char   *start;
    int     last_ch;

#define RETURN(errval, result) { dict->error = errval; return (result); }

    if (msg_verbose)
	msg_info("%s: key %s", myname, key);

    /*
     * Optionally fold the key.
     */
    if (dict->flags & DICT_FLAG_FOLD_MUL) {
	if (dict->fold_buf == 0)
	    dict->fold_buf = vstring_alloc(10);
	vstring_strcpy(dict->fold_buf, key);
	key = lowercase(vstring_str(dict->fold_buf));
    }
    for (tries = 0; /* see below */ ; /* see below */ ) {

	/*
	 * Connect to the server, or use an existing connection.
	 */
	if (dict_tcp->fp != 0 || dict_tcp_connect(dict_tcp) == 0) {

	    /*
	     * Send request and receive response. Both are %XX quoted and
	     * both are terminated by newline. This encoding is convenient
	     * for data that is mostly text.
	     */
	    hex_quote(dict_tcp->hex_buf, key);
	    vstream_fprintf(dict_tcp->fp, "get %s\n", STR(dict_tcp->hex_buf));
	    if (msg_verbose)
		msg_info("%s: send: get %s", myname, STR(dict_tcp->hex_buf));
	    last_ch = vstring_get_nonl_bound(dict_tcp->hex_buf, dict_tcp->fp,
					     DICT_TCP_MAXLEN);
	    if (last_ch == '\n')
		break;

	    /*
	     * Disconnect from the server if it can't talk to us.
	     */
	    if (last_ch < 0)
		msg_warn("read TCP map reply from %s: unexpected EOF (%m)",
			 dict_tcp->dict.name);
	    else
		msg_warn("read TCP map reply from %s: text longer than %d",
			 dict_tcp->dict.name, DICT_TCP_MAXLEN);
	    dict_tcp_disconnect(dict_tcp);
	}

	/*
	 * Try to connect a limited number of times before giving up.
	 */
	if (++tries >= DICT_TCP_MAXTRY)
	    RETURN(DICT_ERR_RETRY, 0);

	/*
	 * Sleep between attempts, instead of hammering the server.
	 */
	sleep(1);
    }
    if (msg_verbose)
	msg_info("%s: recv: %s", myname, STR(dict_tcp->hex_buf));

    /*
     * Check the general reply syntax. If the reply is malformed, disconnect
     * and try again later.
     */
    if (start = STR(dict_tcp->hex_buf),
	!ISDIGIT(start[0]) || !ISDIGIT(start[1])
	|| !ISDIGIT(start[2]) || !ISSPACE(start[3])
	|| !hex_unquote(dict_tcp->raw_buf, start + 4)) {
	msg_warn("read TCP map reply from %s: malformed reply: %.100s",
	       dict_tcp->dict.name, printable(STR(dict_tcp->hex_buf), '_'));
	dict_tcp_disconnect(dict_tcp);
	RETURN(DICT_ERR_RETRY, 0);
    }

    /*
     * Examine the reply status code. If the reply is malformed, disconnect
     * and try again later.
     */
    switch (start[0]) {
    default:
	msg_warn("read TCP map reply from %s: bad status code: %.100s",
	       dict_tcp->dict.name, printable(STR(dict_tcp->hex_buf), '_'));
	dict_tcp_disconnect(dict_tcp);
	RETURN(DICT_ERR_RETRY, 0);
    case '4':
	if (msg_verbose)
	    msg_info("%s: soft error: %s",
		     myname, printable(STR(dict_tcp->hex_buf), '_'));
	dict_tcp_disconnect(dict_tcp);
	RETURN(DICT_ERR_RETRY, 0);
    case '5':
	if (msg_verbose)
	    msg_info("%s: not found: %s",
		     myname, printable(STR(dict_tcp->hex_buf), '_'));
	RETURN(DICT_ERR_NONE, 0);
    case '2':
	if (msg_verbose)
	    msg_info("%s: found: %s",
		     myname, printable(STR(dict_tcp->raw_buf), '_'));
	RETURN(DICT_ERR_NONE, STR(dict_tcp->raw_buf));
    }
}

/* dict_tcp_close - close TCP map */

static void dict_tcp_close(DICT *dict)
{
    DICT_TCP *dict_tcp = (DICT_TCP *) dict;

    if (dict_tcp->fp)
	(void) vstream_fclose(dict_tcp->fp);
    if (dict_tcp->raw_buf)
	vstring_free(dict_tcp->raw_buf);
    if (dict_tcp->hex_buf)
	vstring_free(dict_tcp->hex_buf);
    if (dict->fold_buf)
	vstring_free(dict->fold_buf);
    dict_free(dict);
}

/* dict_tcp_open - open TCP map */

DICT   *dict_tcp_open(const char *map, int open_flags, int dict_flags)
{
    DICT_TCP *dict_tcp;

    /*
     * Sanity checks.
     */
    if (dict_flags & DICT_FLAG_NO_UNAUTH)
	return (dict_surrogate(DICT_TYPE_TCP, map, open_flags, dict_flags,
		     "%s:%s map is not allowed for security sensitive data",
			       DICT_TYPE_TCP, map));
    if (open_flags != O_RDONLY)
	return (dict_surrogate(DICT_TYPE_TCP, map, open_flags, dict_flags,
			       "%s:%s map requires O_RDONLY access mode",
			       DICT_TYPE_TCP, map));

    /*
     * Create the dictionary handle. Do not open the connection until the
     * first request is made.
     */
    dict_tcp = (DICT_TCP *) dict_alloc(DICT_TYPE_TCP, map, sizeof(*dict_tcp));
    dict_tcp->fp = 0;
    dict_tcp->raw_buf = dict_tcp->hex_buf = 0;
    dict_tcp->dict.lookup = dict_tcp_lookup;
    dict_tcp->dict.close = dict_tcp_close;
    dict_tcp->dict.flags = dict_flags | DICT_FLAG_PATTERN;
    if (dict_flags & DICT_FLAG_FOLD_MUL)
	dict_tcp->dict.fold_buf = vstring_alloc(10);

    return (DICT_DEBUG (&dict_tcp->dict));
}
