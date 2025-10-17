/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 6, 2024.
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
#include <sys/time.h>

#ifndef timersub
/* res = a - b */
#define timersub(a, b, res) do { \
	(res)->tv_sec = (a)->tv_sec - (b)->tv_sec; \
	(res)->tv_usec = (a)->tv_usec - (b)->tv_usec; \
	if ((res)->tv_usec < 0) { \
		(res)->tv_sec--; \
		(res)->tv_usec += 1000000; \
	} \
    } while (0)
#endif

#ifdef USE_TLS

/* Utility library. */

#include <msg.h>
#include <iostuff.h>

/* TLS library. */

#define TLS_INTERNAL
#include <tls.h>

/* tls_bio - perform SSL input/output operation with extreme prejudice */

int     tls_bio(int fd, int timeout, TLS_SESS_STATE *TLScontext,
		        int (*hsfunc) (SSL *),
		        int (*rfunc) (SSL *, void *, int),
		        int (*wfunc) (SSL *, const void *, int),
		        void *buf, int num)
{
    const char *myname = "tls_bio";
    int     status;
    int     err;
    int     enable_deadline;
    struct timeval time_left;		/* amount of time left */
    struct timeval time_deadline;	/* time of deadline */
    struct timeval time_now;		/* time after SSL_mumble() call */

    /*
     * Compensation for interface mis-match: With VSTREAMs, timeout <= 0
     * means wait forever; with the read/write_wait() calls below, we need to
     * specify timeout < 0 instead.
     * 
     * Safety: no time limit means no deadline.
     */
    if (timeout <= 0) {
	timeout = -1;
	enable_deadline = 0;
    }

    /*
     * Deadline management is simpler than with VSTREAMs, because we don't
     * need to decrement a per-stream time limit. We just work within the
     * budget that is available for this tls_bio() call.
     */
    else {
	enable_deadline =
	    vstream_fstat(TLScontext->stream, VSTREAM_FLAG_DEADLINE);
	if (enable_deadline) {
	    GETTIMEOFDAY(&time_deadline);
	    time_deadline.tv_sec += timeout;
	}
    }

    /*
     * If necessary, retry the SSL handshake or read/write operation after
     * handling any pending network I/O.
     */
    for (;;) {
	if (hsfunc)
	    status = hsfunc(TLScontext->con);
	else if (rfunc)
	    status = rfunc(TLScontext->con, buf, num);
	else if (wfunc)
	    status = wfunc(TLScontext->con, buf, num);
	else
	    msg_panic("%s: nothing to do here", myname);
	err = SSL_get_error(TLScontext->con, status);

	/*
	 * Correspondence between SSL_ERROR_* error codes and tls_bio_(read,
	 * write, accept, connect, shutdown) return values (for brevity:
	 * retval).
	 * 
	 * SSL_ERROR_NONE corresponds with retval > 0. With SSL_(read, write)
	 * this is the number of plaintext bytes sent or received. With
	 * SSL_(accept, connect, shutdown) this means that the operation was
	 * completed successfully.
	 * 
	 * SSL_ERROR_WANT_(WRITE, READ) start a new loop iteration, or force
	 * (retval = -1, errno = ETIMEDOUT) when the time limit is exceeded.
	 * 
	 * All other SSL_ERROR_* cases correspond with retval <= 0. With
	 * SSL_(read, write, accept, connect) retval == 0 means that the
	 * remote party either closed the network connection or that it
	 * requested TLS shutdown; with SSL_shutdown() retval == 0 means that
	 * our own shutdown request is in progress. With all operations
	 * retval < 0 means that there was an error. In the latter case,
	 * SSL_ERROR_SYSCALL means that error details are returned via the
	 * errno value.
	 * 
	 * Find out if we must retry the operation and/or if there is pending
	 * network I/O.
	 * 
	 * XXX If we're the first to invoke SSL_shutdown(), then the operation
	 * isn't really complete when the call returns. We could hide that
	 * anomaly here and repeat the call.
	 */
	switch (err) {
	case SSL_ERROR_WANT_WRITE:
	case SSL_ERROR_WANT_READ:
	    if (enable_deadline) {
		GETTIMEOFDAY(&time_now);
		timersub(&time_deadline, &time_now, &time_left);
		timeout = time_left.tv_sec + (time_left.tv_usec > 0);
		if (timeout <= 0) {
		    errno = ETIMEDOUT;
		    return (-1);
		}
	    }
	    if (err == SSL_ERROR_WANT_WRITE) {
		if (write_wait(fd, timeout) < 0)
		    return (-1);		/* timeout error */
	    } else {
		if (read_wait(fd, timeout) < 0)
		    return (-1);		/* timeout error */
	    }
	    break;

	    /*
	     * Unhandled cases: SSL_ERROR_WANT_(ACCEPT, CONNECT, X509_LOOKUP)
	     * etc. Historically, Postfix silently treated these as ordinary
	     * I/O errors so we don't really know how common they are. For
	     * now, we just log a warning.
	     */
	default:
	    msg_warn("%s: unexpected SSL_ERROR code %d", myname, err);
	    /* FALLTHROUGH */

	    /*
	     * With tls_timed_read() and tls_timed_write() the caller is the
	     * VSTREAM library module which is unaware of TLS, so we log the
	     * TLS error stack here. In a better world, each VSTREAM I/O
	     * object would provide an error reporting method in addition to
	     * the timed_read and timed_write methods, so that we would not
	     * need to have ad-hoc code like this.
	     */
	case SSL_ERROR_SSL:
	    if (rfunc || wfunc)
		tls_print_errors();
	    /* FALLTHROUGH */
	case SSL_ERROR_ZERO_RETURN:
	case SSL_ERROR_NONE:
	    errno = 0;				/* avoid bogus warnings */
	    /* FALLTHROUGH */
	case SSL_ERROR_SYSCALL:
	    return (status);
	}
    }
}

#endif
