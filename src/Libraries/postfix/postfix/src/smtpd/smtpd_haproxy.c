/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 7, 2024.
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

/* Utility library. */

#include <msg.h>
#include <myaddrinfo.h>
#include <mymalloc.h>
#include <stringops.h>

/* Global library. */

#include <smtp_stream.h>
#include <mail_params.h>
#include <valid_mailhost_addr.h>
#include <haproxy_srvr.h>

/* Application-specific. */

#include <smtpd.h>

/* SLMs. */

#define STR(x)	vstring_str(x)
#define LEN(x)	VSTRING_LEN(x)

/* smtpd_peer_from_haproxy - initialize peer information from haproxy */

int     smtpd_peer_from_haproxy(SMTPD_STATE *state)
{
    const char *myname = "smtpd_peer_from_haproxy";
    MAI_HOSTADDR_STR smtp_client_addr;
    MAI_SERVPORT_STR smtp_client_port;
    MAI_HOSTADDR_STR smtp_server_addr;
    MAI_SERVPORT_STR smtp_server_port;
    const char *proxy_err;
    int     io_err;
    VSTRING *escape_buf;

    /*
     * While reading HAProxy handshake information, don't buffer input beyond
     * the end-of-line. That would break the TLS wrappermode handshake.
     */
    vstream_control(state->client,
		    VSTREAM_CTL_BUFSIZE, 1,
		    VSTREAM_CTL_END);

    /*
     * Note: the haproxy_srvr_parse() routine performs address protocol
     * checks, address and port syntax checks, and converts IPv4-in-IPv6
     * address string syntax (:ffff::1.2.3.4) to IPv4 syntax where permitted
     * by the main.cf:inet_protocols setting, but logs no warnings.
     */
#define ENABLE_DEADLINE	1

    smtp_stream_setup(state->client, var_smtpd_uproxy_tmout, ENABLE_DEADLINE);
    switch (io_err = vstream_setjmp(state->client)) {
    default:
	msg_panic("%s: unhandled I/O error %d", myname, io_err);
    case SMTP_ERR_EOF:
	msg_warn("haproxy read: unexpected EOF");
	return (-1);
    case SMTP_ERR_TIME:
	msg_warn("haproxy read: timeout error");
	return (-1);
    case 0:
	if (smtp_get(state->buffer, state->client, HAPROXY_MAX_LEN,
		     SMTP_GET_FLAG_NONE) != '\n') {
	    msg_warn("haproxy read: line > %d characters", HAPROXY_MAX_LEN);
	    return (-1);
	}
	if ((proxy_err = haproxy_srvr_parse(STR(state->buffer),
				       &smtp_client_addr, &smtp_client_port,
			      &smtp_server_addr, &smtp_server_port)) != 0) {
	    escape_buf = vstring_alloc(HAPROXY_MAX_LEN + 2);
	    escape(escape_buf, STR(state->buffer), LEN(state->buffer));
	    msg_warn("haproxy read: %s: %s", proxy_err, STR(escape_buf));
	    vstring_free(escape_buf);
	    return (-1);
	}
	state->addr = mystrdup(smtp_client_addr.buf);
	if (strrchr(state->addr, ':') != 0) {
	    state->rfc_addr = concatenate(IPV6_COL, state->addr, (char *) 0);
	    state->addr_family = AF_INET6;
	} else {
	    state->rfc_addr = mystrdup(state->addr);
	    state->addr_family = AF_INET;
	}
	state->port = mystrdup(smtp_client_port.buf);

	/*
	 * The Dovecot authentication server needs the server IP address.
	 */
	state->dest_addr = mystrdup(smtp_server_addr.buf);
	state->dest_port = mystrdup(smtp_server_port.buf);

	/*
	 * Enable normal buffering.
	 */
	vstream_control(state->client,
			VSTREAM_CTL_BUFSIZE, VSTREAM_BUFSIZE,
			VSTREAM_CTL_END);
	return (0);
    }
}
