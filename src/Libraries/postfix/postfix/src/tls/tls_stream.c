/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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

#include <iostuff.h>
#include <vstream.h>
#include <msg.h>

/* TLS library. */

#define TLS_INTERNAL
#include <tls.h>

 /*
  * Interface mis-match compensation. The OpenSSL read/write routines return
  * unspecified negative values when an operation fails, while the vstream(3)
  * plaintext timed_read/write() functions follow the convention of UNIX
  * system calls, and return -1 upon error. The macro below makes OpenSSL
  * read/write results consistent with the UNIX system-call convention.
  */
#define NORMALIZED_VSTREAM_RETURN(retval) ((retval) < 0 ? -1 : (retval))

/* tls_timed_read - read content from stream, then TLS decapsulate */

static ssize_t tls_timed_read(int fd, void *buf, size_t len, int timeout,
			              void *context)
{
    const char *myname = "tls_timed_read";
    ssize_t ret;
    TLS_SESS_STATE *TLScontext;

    TLScontext = (TLS_SESS_STATE *) context;
    if (!TLScontext)
	msg_panic("%s: no context", myname);

    ret = tls_bio_read(fd, buf, len, timeout, TLScontext);
    if (ret > 0 && (TLScontext->log_mask & TLS_LOG_ALLPKTS))
	msg_info("Read %ld chars: %.*s",
		 (long) ret, (int) (ret > 40 ? 40 : ret), (char *) buf);
    return (NORMALIZED_VSTREAM_RETURN(ret));
}

/* tls_timed_write - TLS encapsulate content, then write to stream */

static ssize_t tls_timed_write(int fd, void *buf, size_t len, int timeout,
			               void *context)
{
    const char *myname = "tls_timed_write";
    ssize_t ret;
    TLS_SESS_STATE *TLScontext;

    TLScontext = (TLS_SESS_STATE *) context;
    if (!TLScontext)
	msg_panic("%s: no context", myname);

    if (TLScontext->log_mask & TLS_LOG_ALLPKTS)
	msg_info("Write %ld chars: %.*s",
		 (long) len, (int) (len > 40 ? 40 : len), (char *) buf);
    ret = tls_bio_write(fd, buf, len, timeout, TLScontext);
    return (NORMALIZED_VSTREAM_RETURN(ret));
}

/* tls_stream_start - start VSTREAM over TLS */

void    tls_stream_start(VSTREAM *stream, TLS_SESS_STATE *context)
{
    vstream_control(stream,
		    CA_VSTREAM_CTL_READ_FN(tls_timed_read),
		    CA_VSTREAM_CTL_WRITE_FN(tls_timed_write),
		    CA_VSTREAM_CTL_CONTEXT(context),
		    CA_VSTREAM_CTL_END);
}

/* tls_stream_stop - stop VSTREAM over TLS */

void    tls_stream_stop(VSTREAM *stream)
{

    /*
     * Prevent data leakage after TLS is turned off. The Postfix/TLS patch
     * provided null function pointers; we use dummy routines that make less
     * noise when used.
     */
    vstream_control(stream,
		    CA_VSTREAM_CTL_READ_FN(dummy_read),
		    CA_VSTREAM_CTL_WRITE_FN(dummy_write),
		    CA_VSTREAM_CTL_CONTEXT((void *) 0),
		    CA_VSTREAM_CTL_END);
}

#endif
