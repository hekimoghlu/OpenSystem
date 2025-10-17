/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 25, 2024.
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
#include <string.h>

#ifdef STRCASECMP_IN_STRINGS_H
#include <strings.h>
#endif

/* Utility library. */

#include <msg.h>
#include <myaddrinfo.h>
#include <vstream.h>
#include <inet_proto.h>

/* Global library. */

#include <mail_params.h>
#include <haproxy_srvr.h>

/* Application-specific. */

#include <postscreen.h>
#include <postscreen_haproxy.h>

static INET_PROTO_INFO *proto_info;

/* psc_sockaddr_to_hostaddr - transform endpoint address and port to string */

static int psc_sockaddr_to_hostaddr(struct sockaddr *addr_storage,
				            SOCKADDR_SIZE addr_storage_len,
				            MAI_HOSTADDR_STR *addr_buf,
				            MAI_SERVPORT_STR *port_buf,
				            int socktype)
{
    int     aierr;

    if ((aierr = sockaddr_to_hostaddr(addr_storage, addr_storage_len,
				      addr_buf, port_buf, socktype)) == 0
	&& strncasecmp("::ffff:", addr_buf->buf, 7) == 0
	&& strchr((char *) proto_info->sa_family_list, AF_INET) != 0)
	memmove(addr_buf->buf, addr_buf->buf + 7,
		sizeof(addr_buf->buf) - 7);
    return (aierr);
}

/* psc_endpt_local_lookup - look up local system connection information */

static void psc_endpt_local_lookup(VSTREAM *smtp_client_stream,
				           PSC_ENDPT_LOOKUP_FN lookup_done)
{
    struct sockaddr_storage addr_storage;
    SOCKADDR_SIZE addr_storage_len = sizeof(addr_storage);
    int     status;
    MAI_HOSTADDR_STR smtp_client_addr;
    MAI_SERVPORT_STR smtp_client_port;
    MAI_HOSTADDR_STR smtp_server_addr;
    MAI_SERVPORT_STR smtp_server_port;
    int     aierr;

    /*
     * Look up the remote SMTP client address and port.
     */
    if (getpeername(vstream_fileno(smtp_client_stream), (struct sockaddr *)
		    &addr_storage, &addr_storage_len) < 0) {
	msg_warn("getpeername: %m -- dropping this connection");
	status = -1;
    }

    /*
     * Convert the remote SMTP client address and port to printable form for
     * logging and access control.
     */
    else if ((aierr = psc_sockaddr_to_hostaddr(
					  (struct sockaddr *) &addr_storage,
					addr_storage_len, &smtp_client_addr,
				    &smtp_client_port, SOCK_STREAM)) != 0) {
	msg_warn("cannot convert client address/port to string: %s"
		 " -- dropping this connection",
		 MAI_STRERROR(aierr));
	status = -1;
    }

    /*
     * Look up the local SMTP server address and port.
     */
    else if (getsockname(vstream_fileno(smtp_client_stream),
			 (struct sockaddr *) &addr_storage,
			 &addr_storage_len) < 0) {
	msg_warn("getsockname: %m -- dropping this connection");
	status = -1;
    }

    /*
     * Convert the local SMTP server address and port to printable form for
     * logging.
     */
    else if ((aierr = psc_sockaddr_to_hostaddr(
					  (struct sockaddr *) &addr_storage,
					addr_storage_len, &smtp_server_addr,
				    &smtp_server_port, SOCK_STREAM)) != 0) {
	msg_warn("cannot convert server address/port to string: %s"
		 " -- dropping this connection",
		 MAI_STRERROR(aierr));
	status = -1;
    } else {
	status = 0;
    }
    lookup_done(status, smtp_client_stream,
		&smtp_client_addr, &smtp_client_port,
		&smtp_server_addr, &smtp_server_port);
}

 /*
  * Lookup table for available proxy protocols.
  */
typedef struct {
    const char *name;
    void    (*endpt_lookup) (VSTREAM *, PSC_ENDPT_LOOKUP_FN);
} PSC_ENDPT_LOOKUP_INFO;

static const PSC_ENDPT_LOOKUP_INFO psc_endpt_lookup_info[] = {
    NOPROXY_PROTO_NAME, psc_endpt_local_lookup,
    HAPROXY_PROTO_NAME, psc_endpt_haproxy_lookup,
    0,
};

/* psc_endpt_lookup - look up connection endpoint information */

void    psc_endpt_lookup(VSTREAM *smtp_client_stream,
			         PSC_ENDPT_LOOKUP_FN notify)
{
    const PSC_ENDPT_LOOKUP_INFO *pp;

    if (proto_info == 0)
	proto_info = inet_proto_info();

    for (pp = psc_endpt_lookup_info; /* see below */ ; pp++) {
	if (pp->name == 0)
	    msg_fatal("unsupported %s value: %s",
		      VAR_PSC_UPROXY_PROTO, var_psc_uproxy_proto);
	if (strcmp(var_psc_uproxy_proto, pp->name) == 0) {
	    pp->endpt_lookup(smtp_client_stream, notify);
	    return;
	}
    }
}
