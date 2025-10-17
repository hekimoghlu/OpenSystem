/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 24, 2025.
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
#ifdef USE_TLS

/* System library. */

#include <sys_defs.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <connect.h>
#include <stringops.h>
#include <vstring.h>

/* Global library. */

#include <mail_proto.h>
#include <mail_params.h>

/* TLS library-specific. */

#include <tls.h>
#include <tls_proxy.h>

#define TLSPROXY_INIT_TIMEOUT		10

/* SLMs. */

#define STR	vstring_str

/* tls_proxy_open - open negotiations with TLS proxy */

VSTREAM *tls_proxy_open(const char *service, int flags,
			        VSTREAM *peer_stream,
			        const char *peer_addr,
			        const char *peer_port,
			        int timeout)
{
    VSTREAM *tlsproxy_stream;
    int     status;
    int     fd;
    static VSTRING *tlsproxy_service = 0;
    static VSTRING *remote_endpt = 0;

    /*
     * Initialize.
     */
    if (tlsproxy_service == 0) {
	tlsproxy_service = vstring_alloc(20);
	remote_endpt = vstring_alloc(20);
    }

    /*
     * Connect to the tlsproxy(8) daemon.
     */
    vstring_sprintf(tlsproxy_service, "%s/%s", MAIL_CLASS_PRIVATE, service);
    if ((fd = LOCAL_CONNECT(STR(tlsproxy_service), BLOCKING,
			    TLSPROXY_INIT_TIMEOUT)) < 0) {
	msg_warn("connect to %s service: %m", STR(tlsproxy_service));
	return (0);
    }

    /*
     * Initial handshake. Send the data attributes now, and send the client
     * file descriptor in a later transaction.
     * 
     * XXX The formatted endpoint should be a state member. Then, we can
     * simplify all the format strings throughout the program.
     */
    tlsproxy_stream = vstream_fdopen(fd, O_RDWR);
    vstring_sprintf(remote_endpt, "[%s]:%s", peer_addr, peer_port);
    attr_print(tlsproxy_stream, ATTR_FLAG_NONE,
	       SEND_ATTR_STR(MAIL_ATTR_REMOTE_ENDPT, STR(remote_endpt)),
	       SEND_ATTR_INT(MAIL_ATTR_FLAGS, flags),
	       SEND_ATTR_INT(MAIL_ATTR_TIMEOUT, timeout),
	       ATTR_TYPE_END);
    if (vstream_fflush(tlsproxy_stream) != 0) {
	msg_warn("error sending request to %s service: %m",
		 STR(tlsproxy_service));
	vstream_fclose(tlsproxy_stream);
	return (0);
    }

    /*
     * Receive the "TLS is available" indication.
     * 
     * This may seem out of order, but we must have a read transaction between
     * sending the request attributes and sending the SMTP client file
     * descriptor. We can't assume UNIX-domain socket semantics here.
     */
    if (attr_scan(tlsproxy_stream, ATTR_FLAG_STRICT,
		  RECV_ATTR_INT(MAIL_ATTR_STATUS, &status),
		  ATTR_TYPE_END) != 1 || status == 0) {

	/*
	 * The TLS proxy reports that the TLS engine is not available (due to
	 * configuration error, or other causes).
	 */
	msg_warn("%s service role \"%s\" is not available",
		 STR(tlsproxy_service),
		 (flags & TLS_PROXY_FLAG_ROLE_SERVER) ? "server" :
		 (flags & TLS_PROXY_FLAG_ROLE_CLIENT) ? "client" :
		 "bogus role");
	vstream_fclose(tlsproxy_stream);
	return (0);
    }

    /*
     * Send the remote SMTP client file descriptor.
     */
    if (LOCAL_SEND_FD(vstream_fileno(tlsproxy_stream),
		      vstream_fileno(peer_stream)) < 0) {

	/*
	 * Some error: drop the TLS proxy stream.
	 */
	msg_warn("sending file handle to %s service: %m",
		 STR(tlsproxy_service));
	vstream_fclose(tlsproxy_stream);
	return (0);
    }
    return (tlsproxy_stream);
}

/* tls_proxy_context_receive - receive TLS session object from tlsproxy(8) */

TLS_SESS_STATE *tls_proxy_context_receive(VSTREAM *proxy_stream)
{
    TLS_SESS_STATE *tls_context;

    tls_context = (TLS_SESS_STATE *) mymalloc(sizeof(*tls_context));

    if (attr_scan(proxy_stream, ATTR_FLAG_STRICT,
	       RECV_ATTR_FUNC(tls_proxy_context_scan, (void *) tls_context),
		  ATTR_TYPE_END) != 1) {
	tls_proxy_context_free(tls_context);
	return (0);
    } else {
	return (tls_context);
    }
}

/* tls_proxy_context_free - destroy object from tls_proxy_context_receive() */

void    tls_proxy_context_free(TLS_SESS_STATE *tls_context)
{
    if (tls_context->peer_CN)
	myfree(tls_context->peer_CN);
    if (tls_context->issuer_CN)
	myfree(tls_context->issuer_CN);
    if (tls_context->peer_cert_fprint)
	myfree(tls_context->peer_cert_fprint);
    if (tls_context->protocol)
	myfree((void *) tls_context->protocol);
    if (tls_context->cipher_name)
	myfree((void *) tls_context->cipher_name);
    myfree((void *) tls_context);
}

#endif
