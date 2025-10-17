/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
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

#ifdef USE_TLS

/* Utility library. */

#include <vstream.h>
#include <msg.h>
#include <mymalloc.h>

/* TLS library. */

#define TLS_INTERNAL
#include <tls.h>

/* Application-specific. */

#define STR	vstring_str

/* tls_session_stop - shut down the TLS connection and reset state */

void    tls_session_stop(TLS_APPL_STATE *unused_ctx, VSTREAM *stream, int timeout,
			         int failure, TLS_SESS_STATE *TLScontext)
{
    const char *myname = "tls_session_stop";
    int     retval;

    /*
     * Sanity check.
     */
    if (TLScontext == 0)
	msg_panic("%s: stream has no active TLS context", myname);

    /*
     * Perform SSL_shutdown() twice, as the first attempt will send out the
     * shutdown alert but it will not wait for the peer's shutdown alert.
     * Therefore, when we are the first party to send the alert, we must call
     * SSL_shutdown() again. On failure we don't want to resume the session,
     * so we will not perform SSL_shutdown() and the session will be removed
     * as being bad.
     */
    if (!failure) {
	retval = tls_bio_shutdown(vstream_fileno(stream), timeout, TLScontext);
	if (retval == 0)
	    tls_bio_shutdown(vstream_fileno(stream), timeout, TLScontext);
    }
    tls_free_context(TLScontext);
    tls_stream_stop(stream);
}

/* tls_session_passivate - passivate SSL_SESSION object */

VSTRING *tls_session_passivate(SSL_SESSION *session)
{
    const char *myname = "tls_session_passivate";
    int     estimate;
    int     actual_size;
    VSTRING *session_data;
    unsigned char *ptr;

    /*
     * First, find out how much memory is needed for the passivated
     * SSL_SESSION object.
     */
    estimate = i2d_SSL_SESSION(session, (unsigned char **) 0);
    if (estimate <= 0) {
	msg_warn("%s: i2d_SSL_SESSION failed: unable to cache session", myname);
	return (0);
    }

    /*
     * Passivate the SSL_SESSION object. The use of a VSTRING is slightly
     * wasteful but is convenient to combine data and length.
     */
    session_data = vstring_alloc(estimate);
    ptr = (unsigned char *) STR(session_data);
    actual_size = i2d_SSL_SESSION(session, &ptr);
    if (actual_size != estimate) {
	msg_warn("%s: i2d_SSL_SESSION failed: unable to cache session", myname);
	vstring_free(session_data);
	return (0);
    }
    VSTRING_AT_OFFSET(session_data, actual_size);	/* XXX not public */

    return (session_data);
}

/* tls_session_activate - activate passivated session */

SSL_SESSION *tls_session_activate(const char *session_data, int session_data_len)
{
#if (OPENSSL_VERSION_NUMBER < 0x0090707fL)
#define BOGUS_CONST
#else
#define BOGUS_CONST const
#endif
    SSL_SESSION *session;
    BOGUS_CONST unsigned char *ptr;

    /*
     * Activate the SSL_SESSION object.
     */
    ptr = (BOGUS_CONST unsigned char *) session_data;
    session = d2i_SSL_SESSION((SSL_SESSION **) 0, &ptr, session_data_len);
    if (!session)
	tls_print_errors();

    return (session);
}

#endif
