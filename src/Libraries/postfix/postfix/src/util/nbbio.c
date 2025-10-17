/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 8, 2024.
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
/*
  * System library.
  */
#include <sys_defs.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>			/* memmove() */

 /*
  * Utility library.
  */
#include <mymalloc.h>
#include <msg.h>
#include <events.h>
#include <nbbio.h>

/* nbbio_event - non-blocking event handler */

static void nbbio_event(int event, void *context)
{
    const char *myname = "nbbio_event";
    NBBIO  *np = (NBBIO *) context;
    ssize_t count;

    switch (event) {

	/*
	 * Read data into the read buffer. Leave it up to the application to
	 * drain the buffer until it is empty.
	 */
    case EVENT_READ:
	if (np->read_pend == np->bufsize)
	    msg_panic("%s: socket fd=%d: read buffer is full",
		      myname, np->fd);
	if (np->read_pend < 0 || np->read_pend > np->bufsize)
	    msg_panic("%s: socket fd=%d: bad pending read count %ld",
		      myname, np->fd, (long) np->read_pend);
	count = read(np->fd, np->read_buf + np->read_pend,
		     np->bufsize - np->read_pend);
	if (count > 0) {
	    np->read_pend += count;
	    if (msg_verbose)
		msg_info("%s: read %ld on %s fd=%d",
			 myname, (long) count, np->label, np->fd);
	} else if (count == 0) {
	    np->flags |= NBBIO_FLAG_EOF;
	    if (msg_verbose)
		msg_info("%s: read EOF on %s fd=%d",
			 myname, np->label, np->fd);
	} else {
	    if (errno == EAGAIN)
		msg_warn("%s: read() returns EAGAIN on readable descriptor",
			 myname);
	    np->flags |= NBBIO_FLAG_ERROR;
	    if (msg_verbose)
		msg_info("%s: read %s fd=%d: %m", myname, np->label, np->fd);
	}
	break;

	/*
	 * Drain data from the output buffer.  Notify the application
	 * whenever some bytes are written.
	 * 
	 * XXX Enforce a total time limit to ensure liveness when a hostile
	 * receiver sets a very small TCP window size.
	 */
    case EVENT_WRITE:
	if (np->write_pend == 0)
	    msg_panic("%s: socket fd=%d: empty write buffer", myname, np->fd);
	if (np->write_pend < 0 || np->write_pend > np->bufsize)
	    msg_panic("%s: socket fd=%d: bad pending write count %ld",
		      myname, np->fd, (long) np->write_pend);
	count = write(np->fd, np->write_buf, np->write_pend);
	if (count > 0) {
	    np->write_pend -= count;
	    if (np->write_pend > 0)
		memmove(np->write_buf, np->write_buf + count, np->write_pend);
	} else {
	    if (errno == EAGAIN)
		msg_warn("%s: write() returns EAGAIN on writable descriptor",
			 myname);
	    np->flags |= NBBIO_FLAG_ERROR;
	    if (msg_verbose)
		msg_info("%s: write %s fd=%d: %m", myname, np->label, np->fd);
	}
	break;

	/*
	 * Something bad happened.
	 */
    case EVENT_XCPT:
	np->flags |= NBBIO_FLAG_ERROR;
	if (msg_verbose)
	    msg_info("%s: error on %s fd=%d: %m", myname, np->label, np->fd);
	break;

	/*
	 * Something good didn't happen.
	 */
    case EVENT_TIME:
	np->flags |= NBBIO_FLAG_TIMEOUT;
	if (msg_verbose)
	    msg_info("%s: %s timeout on %s fd=%d",
		     myname, NBBIO_OP_NAME(np), np->label, np->fd);
	break;

    default:
	msg_panic("%s: unknown event %d", myname, event);
    }

    /*
     * Application notification. The application will check for any error
     * flags, copy application data from or to our buffer pair, and decide
     * what I/O happens next.
     */
    np->action(event, np->context);
}

/* nbbio_enable_read - enable reading from socket into buffer */

void    nbbio_enable_read(NBBIO *np, int timeout)
{
    const char *myname = "nbbio_enable_read";

    /*
     * Sanity checks.
     */
    if (np->flags & NBBIO_MASK_ACTIVE)
	msg_panic("%s: socket fd=%d is enabled for %s",
		  myname, np->fd, NBBIO_OP_NAME(np));
    if (timeout <= 0)
	msg_panic("%s: socket fd=%d: bad timeout %d",
		  myname, np->fd, timeout);
    if (np->read_pend >= np->bufsize)
	msg_panic("%s: socket fd=%d: read buffer is full",
		  myname, np->fd);

    /*
     * Enable events.
     */
    event_enable_read(np->fd, nbbio_event, (void *) np);
    event_request_timer(nbbio_event, (void *) np, timeout);
    np->flags |= NBBIO_FLAG_READ;
}

/* nbbio_enable_write - enable writing from buffer to socket */

void    nbbio_enable_write(NBBIO *np, int timeout)
{
    const char *myname = "nbbio_enable_write";

    /*
     * Sanity checks.
     */
    if (np->flags & NBBIO_MASK_ACTIVE)
	msg_panic("%s: socket fd=%d is enabled for %s",
		  myname, np->fd, NBBIO_OP_NAME(np));
    if (timeout <= 0)
	msg_panic("%s: socket fd=%d bad timeout %d",
		  myname, np->fd, timeout);
    if (np->write_pend <= 0)
	msg_panic("%s: socket fd=%d: empty write buffer",
		  myname, np->fd);

    /*
     * Enable events.
     */
    event_enable_write(np->fd, nbbio_event, (void *) np);
    event_request_timer(nbbio_event, (void *) np, timeout);
    np->flags |= NBBIO_FLAG_WRITE;
}

/* nbbio_disable_readwrite - disable read/write/timer events */

void    nbbio_disable_readwrite(NBBIO *np)
{
    np->flags &= ~NBBIO_MASK_ACTIVE;
    event_disable_readwrite(np->fd);
    event_cancel_timer(nbbio_event, (void *) np);
}

/* nbbio_slumber - disable read/write events, keep timer */

void    nbbio_slumber(NBBIO *np, int timeout)
{
    np->flags &= ~NBBIO_MASK_ACTIVE;
    event_disable_readwrite(np->fd);
    event_request_timer(nbbio_event, (void *) np, timeout);
}

/* nbbio_create - create socket buffer */

NBBIO  *nbbio_create(int fd, ssize_t bufsize, const char *label,
		             NBBIO_ACTION action, void *context)
{
    NBBIO  *np;

    /*
     * Sanity checks.
     */
    if (fd < 0)
	msg_panic("nbbio_create: bad file descriptor: %d", fd);
    if (bufsize <= 0)
	msg_panic("nbbio_create: bad buffer size: %ld", (long) bufsize);

    /*
     * Create a new buffer pair.
     */
    np = (NBBIO *) mymalloc(sizeof(*np));
    np->fd = fd;
    np->bufsize = bufsize;
    np->label = mystrdup(label);
    np->action = action;
    np->context = context;
    np->flags = 0;

    np->read_buf = mymalloc(bufsize);
    np->read_pend = 0;

    np->write_buf = mymalloc(bufsize);
    np->write_pend = 0;

    return (np);
}

/* nbbio_free - destroy socket buffer */

void    nbbio_free(NBBIO *np)
{
    nbbio_disable_readwrite(np);
    (void) close(np->fd);
    myfree(np->label);
    myfree(np->read_buf);
    myfree(np->write_buf);
    myfree((void *) np);
}
