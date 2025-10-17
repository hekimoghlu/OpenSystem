/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 22, 2022.
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
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <events.h>
#include <myaddrinfo.h>
#include <vstream.h>
#include <vstring.h>
#include <stringops.h>

/* Global library. */

#include <haproxy_srvr.h>
#include <mail_params.h>

/* Application-specific. */

#include <postscreen.h>
#include <postscreen_haproxy.h>

 /*
  * Per-session state.
  */
typedef struct {
    VSTREAM *stream;
    PSC_ENDPT_LOOKUP_FN notify;
    VSTRING *buffer;
} PSC_HAPROXY_STATE;

/* psc_endpt_haproxy_event - read or time event */

static void psc_endpt_haproxy_event(int event, void *context)
{
    const char *myname = "psc_endpt_haproxy_event";
    PSC_HAPROXY_STATE *state = (PSC_HAPROXY_STATE *) context;
    int     status = 0;
    MAI_HOSTADDR_STR smtp_client_addr;
    MAI_SERVPORT_STR smtp_client_port;
    MAI_HOSTADDR_STR smtp_server_addr;
    MAI_SERVPORT_STR smtp_server_port;
    int     last_char = 0;
    const char *err;
    VSTRING *escape_buf;
    char    read_buf[HAPROXY_MAX_LEN];
    ssize_t read_len;
    char   *cp;

    /*
     * We must not read(2) past the <CR><LF> that terminates the haproxy
     * line. For efficiency reasons we read the entire haproxy line in one
     * read(2) call when we know that the line is unfragmented. In the rare
     * case that the line is fragmented, we fall back and read(2) it one
     * character at a time.
     */
    switch (event) {
    case EVENT_TIME:
	msg_warn("haproxy read: time limit exceeded");
	status = -1;
	break;
    case EVENT_READ:
	/* Determine the initial VSTREAM read(2) buffer size. */
	if (VSTRING_LEN(state->buffer) == 0) {
	    if ((read_len = recv(vstream_fileno(state->stream),
			      read_buf, sizeof(read_buf) - 1, MSG_PEEK)) > 0
		&& ((cp = memchr(read_buf, '\n', read_len)) != 0)) {
		read_len = cp - read_buf + 1;
	    } else {
		read_len = 1;
	    }
	    vstream_control(state->stream, CA_VSTREAM_CTL_BUFSIZE(read_len),
			    CA_VSTREAM_CTL_END);
	}
	/* Drain the VSTREAM buffer, otherwise this pseudo-thread will hang. */
	do {
	    if ((last_char = VSTREAM_GETC(state->stream)) == VSTREAM_EOF) {
		if (vstream_ferror(state->stream))
		    msg_warn("haproxy read: %m");
		else
		    msg_warn("haproxy read: lost connection");
		status = -1;
		break;
	    }
	    if (VSTRING_LEN(state->buffer) >= HAPROXY_MAX_LEN) {
		msg_warn("haproxy read: line too long");
		status = -1;
		break;
	    }
	    VSTRING_ADDCH(state->buffer, last_char);
	} while (vstream_peek(state->stream) > 0);
	break;
    }

    /*
     * Parse the haproxy line. Note: the haproxy_srvr_parse() routine
     * performs address protocol checks, address and port syntax checks, and
     * converts IPv4-in-IPv6 address string syntax (:ffff::1.2.3.4) to IPv4
     * syntax where permitted by the main.cf:inet_protocols setting.
     */
    if (status == 0 && last_char == '\n') {
	VSTRING_TERMINATE(state->buffer);
	if ((err = haproxy_srvr_parse(vstring_str(state->buffer),
				      &smtp_client_addr, &smtp_client_port,
			      &smtp_server_addr, &smtp_server_port)) != 0) {
	    escape_buf = vstring_alloc(HAPROXY_MAX_LEN + 2);
	    escape(escape_buf, vstring_str(state->buffer),
		   VSTRING_LEN(state->buffer));
	    msg_warn("haproxy read: %s: %s", err, vstring_str(escape_buf));
	    status = -1;
	    vstring_free(escape_buf);
	}
    }

    /*
     * Are we done yet?
     */
    if (status < 0 || last_char == '\n') {
	PSC_CLEAR_EVENT_REQUEST(vstream_fileno(state->stream),
				psc_endpt_haproxy_event, context);
	vstream_control(state->stream,
			CA_VSTREAM_CTL_BUFSIZE(VSTREAM_BUFSIZE),
			CA_VSTREAM_CTL_END);
	state->notify(status, state->stream,
		      &smtp_client_addr, &smtp_client_port,
		      &smtp_server_addr, &smtp_server_port);
	/* Note: the stream may be closed at this point. */
	vstring_free(state->buffer);
	myfree((void *) state);
    }
}

/* psc_endpt_haproxy_lookup - event-driven haproxy client */

void    psc_endpt_haproxy_lookup(VSTREAM *stream,
				         PSC_ENDPT_LOOKUP_FN notify)
{
    const char *myname = "psc_endpt_haproxy_lookup";
    PSC_HAPROXY_STATE *state;

    /*
     * Prepare the per-session state. XXX To improve overload behavior,
     * maintain a pool of these so that we can reduce memory allocator
     * activity.
     */
    state = (PSC_HAPROXY_STATE *) mymalloc(sizeof(*state));
    state->stream = stream;
    state->notify = notify;
    state->buffer = vstring_alloc(100);

    /*
     * Read the haproxy line.
     */
    PSC_READ_EVENT_REQUEST(vstream_fileno(stream), psc_endpt_haproxy_event,
			   (void *) state, var_psc_uproxy_tmout);
}
