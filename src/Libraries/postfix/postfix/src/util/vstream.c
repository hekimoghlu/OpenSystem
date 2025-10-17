/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 26, 2025.
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
#include <stdlib.h>			/* 44BSD stdarg.h uses abort() */
#include <stdarg.h>
#include <stddef.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <errno.h>
#include <string.h>

/* Utility library. */

#include "mymalloc.h"
#include "msg.h"
#include "vbuf_print.h"
#include "iostuff.h"
#include "vstring.h"
#include "vstream.h"

/* Application-specific. */

 /*
  * Forward declarations.
  */
static int vstream_buf_get_ready(VBUF *);
static int vstream_buf_put_ready(VBUF *);
static int vstream_buf_space(VBUF *, ssize_t);

 /*
  * Initialization of the three pre-defined streams. Pre-allocate a static
  * I/O buffer for the standard error stream, so that the error handler can
  * produce a diagnostic even when memory allocation fails.
  */
static unsigned char vstream_fstd_buf[VSTREAM_BUFSIZE];

VSTREAM vstream_fstd[] = {
    {{
	    0,				/* flags */
	    0, 0, 0, 0,			/* buffer */
	    vstream_buf_get_ready, vstream_buf_put_ready, vstream_buf_space,
    }, STDIN_FILENO, (VSTREAM_RW_FN) timed_read, (VSTREAM_RW_FN) timed_write,
    0,},
    {{
	    0,				/* flags */
	    0, 0, 0, 0,			/* buffer */
	    vstream_buf_get_ready, vstream_buf_put_ready, vstream_buf_space,
    }, STDOUT_FILENO, (VSTREAM_RW_FN) timed_read, (VSTREAM_RW_FN) timed_write,
    0,},
    {{
	    VBUF_FLAG_FIXED | VSTREAM_FLAG_WRITE,
	    vstream_fstd_buf, VSTREAM_BUFSIZE, VSTREAM_BUFSIZE, vstream_fstd_buf,
	    vstream_buf_get_ready, vstream_buf_put_ready, vstream_buf_space,
    }, STDERR_FILENO, (VSTREAM_RW_FN) timed_read, (VSTREAM_RW_FN) timed_write,
    VSTREAM_BUFSIZE,},
};

#define VSTREAM_STATIC(v) ((v) >= VSTREAM_IN && (v) <= VSTREAM_ERR)

 /*
  * A bunch of macros to make some expressions more readable. XXX We're
  * assuming that O_RDONLY == 0, O_WRONLY == 1, O_RDWR == 2.
  */
#define VSTREAM_ACC_MASK(f)	((f) & (O_APPEND | O_WRONLY | O_RDWR))

#define VSTREAM_CAN_READ(f)	(VSTREAM_ACC_MASK(f) == O_RDONLY \
				|| VSTREAM_ACC_MASK(f) == O_RDWR)
#define VSTREAM_CAN_WRITE(f)	(VSTREAM_ACC_MASK(f) & O_WRONLY \
				|| VSTREAM_ACC_MASK(f) & O_RDWR \
				|| VSTREAM_ACC_MASK(f) & O_APPEND)

#define VSTREAM_BUF_COUNT(bp, n) \
	((bp)->flags & VSTREAM_FLAG_READ ? -(n) : (n))

#define VSTREAM_BUF_AT_START(bp) { \
	(bp)->cnt = VSTREAM_BUF_COUNT((bp), (bp)->len); \
	(bp)->ptr = (bp)->data; \
    }

#define VSTREAM_BUF_AT_OFFSET(bp, offset) { \
	(bp)->ptr = (bp)->data + (offset); \
	(bp)->cnt = VSTREAM_BUF_COUNT(bp, (bp)->len - (offset)); \
    }

#define VSTREAM_BUF_AT_END(bp) { \
	(bp)->cnt = 0; \
	(bp)->ptr = (bp)->data + (bp)->len; \
    }

#define VSTREAM_BUF_ZERO(bp) { \
	(bp)->flags = 0; \
	(bp)->data = (bp)->ptr = 0; \
	(bp)->len = (bp)->cnt = 0; \
    }

#define VSTREAM_BUF_ACTIONS(bp, get_action, put_action, space_action) { \
	(bp)->get_ready = (get_action); \
	(bp)->put_ready = (put_action); \
	(bp)->space = (space_action); \
    }

#define VSTREAM_SAVE_STATE(stream, buffer, filedes) { \
	stream->buffer = stream->buf; \
	stream->filedes = stream->fd; \
    }

#define VSTREAM_RESTORE_STATE(stream, buffer, filedes) do { \
	stream->buffer.flags = stream->buf.flags; \
	stream->buf = stream->buffer; \
	stream->fd = stream->filedes; \
    } while(0)

#define VSTREAM_FORK_STATE(stream, buffer, filedes) { \
	stream->buffer = stream->buf; \
	stream->filedes = stream->fd; \
	stream->buffer.data = stream->buffer.ptr = 0; \
	stream->buffer.len = stream->buffer.cnt = 0; \
	stream->buffer.flags &= ~VSTREAM_FLAG_FIXED; \
    };

#define VSTREAM_FLAG_READ_DOUBLE (VSTREAM_FLAG_READ | VSTREAM_FLAG_DOUBLE)
#define VSTREAM_FLAG_WRITE_DOUBLE (VSTREAM_FLAG_WRITE | VSTREAM_FLAG_DOUBLE)

#define VSTREAM_FFLUSH_SOME(stream) \
	vstream_fflush_some((stream), (stream)->buf.len - (stream)->buf.cnt)

/* Note: this does not change a negative result into a zero result. */
#define VSTREAM_SUB_TIME(x, y, z) \
    do { \
	(x).tv_sec = (y).tv_sec - (z).tv_sec; \
	(x).tv_usec = (y).tv_usec - (z).tv_usec; \
	while ((x).tv_usec < 0) { \
	    (x).tv_usec += 1000000; \
	    (x).tv_sec -= 1; \
	} \
	while ((x).tv_usec >= 1000000) { \
	    (x).tv_usec -= 1000000; \
	    (x).tv_sec += 1; \
	} \
    } while (0)

/* vstream_buf_init - initialize buffer */

static void vstream_buf_init(VBUF *bp, int flags)
{

    /*
     * Initialize the buffer such that the first data access triggers a
     * buffer boundary action.
     */
    VSTREAM_BUF_ZERO(bp);
    VSTREAM_BUF_ACTIONS(bp,
			VSTREAM_CAN_READ(flags) ? vstream_buf_get_ready : 0,
			VSTREAM_CAN_WRITE(flags) ? vstream_buf_put_ready : 0,
			vstream_buf_space);
}

/* vstream_buf_alloc - allocate buffer memory */

static void vstream_buf_alloc(VBUF *bp, ssize_t len)
{
    VSTREAM *stream = VBUF_TO_APPL(bp, VSTREAM, buf);
    ssize_t used = bp->ptr - bp->data;
    const char *myname = "vstream_buf_alloc";

    if (len < bp->len)
	msg_panic("%s: attempt to shrink buffer", myname);
    if (bp->flags & VSTREAM_FLAG_FIXED)
	msg_panic("%s: unable to extend fixed-size buffer", myname);

    /*
     * Late buffer allocation allows the user to override the default policy.
     * If a buffer already exists, allow for the presence of (output) data.
     */
    bp->data = (unsigned char *)
	(bp->data ? myrealloc((void *) bp->data, len) : mymalloc(len));
    bp->len = len;
    if (bp->flags & VSTREAM_FLAG_READ) {
	bp->ptr = bp->data + used;
	if (bp->flags & VSTREAM_FLAG_DOUBLE)
	    VSTREAM_SAVE_STATE(stream, read_buf, read_fd);
    } else {
	VSTREAM_BUF_AT_OFFSET(bp, used);
	if (bp->flags & VSTREAM_FLAG_DOUBLE)
	    VSTREAM_SAVE_STATE(stream, write_buf, write_fd);
    }
}

/* vstream_buf_wipe - reset buffer to initial state */

static void vstream_buf_wipe(VBUF *bp)
{
    if ((bp->flags & VBUF_FLAG_FIXED) == 0 && bp->data)
	myfree((void *) bp->data);
    VSTREAM_BUF_ZERO(bp);
    VSTREAM_BUF_ACTIONS(bp, 0, 0, 0);
}

/* vstream_fflush_some - flush some buffered data */

static int vstream_fflush_some(VSTREAM *stream, ssize_t to_flush)
{
    const char *myname = "vstream_fflush_some";
    VBUF   *bp = &stream->buf;
    ssize_t used;
    ssize_t left_over;
    void   *data;
    ssize_t len;
    ssize_t n;
    int     timeout;
    struct timeval before;
    struct timeval elapsed;

    /*
     * Sanity checks. It is illegal to flush a read-only stream. Otherwise,
     * if there is buffered input, discard the input. If there is buffered
     * output, require that the amount to flush is larger than the amount to
     * keep, so that we can memcpy() the residue.
     */
    if (bp->put_ready == 0)
	msg_panic("%s: read-only stream", myname);
    switch (bp->flags & (VSTREAM_FLAG_WRITE | VSTREAM_FLAG_READ)) {
    case VSTREAM_FLAG_READ:			/* discard input */
	VSTREAM_BUF_AT_END(bp);
	/* FALLTHROUGH */
    case 0:					/* flush after seek? */
	return ((bp->flags & VSTREAM_FLAG_ERR) ? VSTREAM_EOF : 0);
    case VSTREAM_FLAG_WRITE:			/* output buffered */
	break;
    case VSTREAM_FLAG_WRITE | VSTREAM_FLAG_READ:
	msg_panic("%s: read/write stream", myname);
    }
    used = bp->len - bp->cnt;
    left_over = used - to_flush;

    if (msg_verbose > 2 && stream != VSTREAM_ERR)
	msg_info("%s: fd %d flush %ld", myname, stream->fd, (long) to_flush);
    if (to_flush < 0 || left_over < 0)
	msg_panic("%s: bad to_flush %ld", myname, (long) to_flush);
    if (to_flush < left_over)
	msg_panic("%s: to_flush < left_over", myname);
    if (to_flush == 0)
	return ((bp->flags & VSTREAM_FLAG_ERR) ? VSTREAM_EOF : 0);
    if (bp->flags & VSTREAM_FLAG_ERR)
	return (VSTREAM_EOF);

    /*
     * When flushing a buffer, allow for partial writes. These can happen
     * while talking to a network. Update the cached file seek position, if
     * any.
     * 
     * When deadlines are enabled, we count the elapsed time for each write
     * operation instead of simply comparing the time-of-day clock with a
     * per-stream deadline. The latter could result in anomalies when an
     * application does lengthy processing between write operations. Keep in
     * mind that a receiver may not be able to keep up when a sender suddenly
     * floods it with a lot of data as it tries to catch up with a deadline.
     */
    for (data = (void *) bp->data, len = to_flush; len > 0; len -= n, data += n) {
	if (bp->flags & VSTREAM_FLAG_DEADLINE) {
	    timeout = stream->time_limit.tv_sec + (stream->time_limit.tv_usec > 0);
	    if (timeout <= 0) {
		bp->flags |= (VSTREAM_FLAG_WR_ERR | VSTREAM_FLAG_WR_TIMEOUT);
		errno = ETIMEDOUT;
		return (VSTREAM_EOF);
	    }
	    if (len == to_flush)
		GETTIMEOFDAY(&before);
	    else
		before = stream->iotime;
	} else
	    timeout = stream->timeout;
	if ((n = stream->write_fn(stream->fd, data, len, timeout, stream->context)) <= 0) {
	    bp->flags |= VSTREAM_FLAG_WR_ERR;
	    if (errno == ETIMEDOUT) {
		bp->flags |= VSTREAM_FLAG_WR_TIMEOUT;
		stream->time_limit.tv_sec = stream->time_limit.tv_usec = 0;
	    }
	    return (VSTREAM_EOF);
	}
	if (timeout) {
	    GETTIMEOFDAY(&stream->iotime);
	    if (bp->flags & VSTREAM_FLAG_DEADLINE) {
		VSTREAM_SUB_TIME(elapsed, stream->iotime, before);
		VSTREAM_SUB_TIME(stream->time_limit, stream->time_limit, elapsed);
	    }
	}
	if (msg_verbose > 2 && stream != VSTREAM_ERR && n != to_flush)
	    msg_info("%s: %d flushed %ld/%ld", myname, stream->fd,
		     (long) n, (long) to_flush);
    }
    if (bp->flags & VSTREAM_FLAG_SEEK)
	stream->offset += to_flush;

    /*
     * Allow for partial buffer flush requests. We use memcpy() for reasons
     * of portability to pre-ANSI environments (SunOS 4.x or Ultrix 4.x :-).
     * This is OK because we have already verified that the to_flush count is
     * larger than the left_over count.
     */
    if (left_over > 0)
	memcpy(bp->data, bp->data + to_flush, left_over);
    bp->cnt += to_flush;
    bp->ptr -= to_flush;
    return ((bp->flags & VSTREAM_FLAG_ERR) ? VSTREAM_EOF : 0);
}

/* vstream_fflush_delayed - delayed stream flush for double-buffered stream */

static int vstream_fflush_delayed(VSTREAM *stream)
{
    int     status;

    /*
     * Sanity check.
     */
    if ((stream->buf.flags & VSTREAM_FLAG_READ_DOUBLE) != VSTREAM_FLAG_READ_DOUBLE)
	msg_panic("vstream_fflush_delayed: bad flags");

    /*
     * Temporarily swap buffers and flush unwritten data. This may seem like
     * a lot of work, but it's peanuts compared to the write(2) call that we
     * already have avoided. For example, delayed flush is never used on a
     * non-pipelined SMTP connection.
     */
    stream->buf.flags &= ~VSTREAM_FLAG_READ;
    VSTREAM_SAVE_STATE(stream, read_buf, read_fd);
    stream->buf.flags |= VSTREAM_FLAG_WRITE;
    VSTREAM_RESTORE_STATE(stream, write_buf, write_fd);

    status = VSTREAM_FFLUSH_SOME(stream);

    stream->buf.flags &= ~VSTREAM_FLAG_WRITE;
    VSTREAM_SAVE_STATE(stream, write_buf, write_fd);
    stream->buf.flags |= VSTREAM_FLAG_READ;
    VSTREAM_RESTORE_STATE(stream, read_buf, read_fd);

    return (status);
}

/* vstream_buf_get_ready - vbuf callback to make buffer ready for reading */

static int vstream_buf_get_ready(VBUF *bp)
{
    VSTREAM *stream = VBUF_TO_APPL(bp, VSTREAM, buf);
    const char *myname = "vstream_buf_get_ready";
    ssize_t n;
    struct timeval before;
    struct timeval elapsed;
    int     timeout;

    /*
     * Detect a change of I/O direction or position. If so, flush any
     * unwritten output immediately when the stream is single-buffered, or
     * when the stream is double-buffered and the read buffer is empty.
     */
    switch (bp->flags & (VSTREAM_FLAG_WRITE | VSTREAM_FLAG_READ)) {
    case VSTREAM_FLAG_WRITE:			/* change direction */
	if (bp->ptr > bp->data)
	    if ((bp->flags & VSTREAM_FLAG_DOUBLE) == 0
		|| stream->read_buf.cnt >= 0)
		if (VSTREAM_FFLUSH_SOME(stream))
		    return (VSTREAM_EOF);
	bp->flags &= ~VSTREAM_FLAG_WRITE;
	if (bp->flags & VSTREAM_FLAG_DOUBLE)
	    VSTREAM_SAVE_STATE(stream, write_buf, write_fd);
	/* FALLTHROUGH */
    case 0:					/* change position */
	bp->flags |= VSTREAM_FLAG_READ;
	if (bp->flags & VSTREAM_FLAG_DOUBLE) {
	    VSTREAM_RESTORE_STATE(stream, read_buf, read_fd);
	    if (bp->cnt < 0)
		return (0);
	}
	/* FALLTHROUGH */
    case VSTREAM_FLAG_READ:			/* no change */
	break;
    case VSTREAM_FLAG_WRITE | VSTREAM_FLAG_READ:
	msg_panic("%s: read/write stream", myname);
    }

    /*
     * If this is the first GET operation, allocate a buffer. Late buffer
     * allocation gives the application a chance to override the default
     * buffering policy.
     * 
     * XXX Subtle code to set the preferred buffer size as late as possible.
     */
    if (stream->req_bufsize == 0)
	stream->req_bufsize = VSTREAM_BUFSIZE;
    if (bp->len < stream->req_bufsize)
	vstream_buf_alloc(bp, stream->req_bufsize);

    /*
     * If the stream is double-buffered and the write buffer is not empty,
     * this is the time to flush the write buffer. Delayed flushes reduce
     * system call overhead, and on TCP sockets, avoid triggering Nagle's
     * algorithm.
     */
    if ((bp->flags & VSTREAM_FLAG_DOUBLE)
	&& stream->write_buf.len > stream->write_buf.cnt)
	if (vstream_fflush_delayed(stream))
	    return (VSTREAM_EOF);

    /*
     * Did we receive an EOF indication?
     */
    if (bp->flags & VSTREAM_FLAG_EOF)
	return (VSTREAM_EOF);

    /*
     * Fill the buffer with as much data as we can handle, or with as much
     * data as is available right now, whichever is less. Update the cached
     * file seek position, if any.
     * 
     * When deadlines are enabled, we count the elapsed time for each read
     * operation instead of simply comparing the time-of-day clock with a
     * per-stream deadline. The latter could result in anomalies when an
     * application does lengthy processing between read operations. Keep in
     * mind that a sender may get blocked, and may not be able to keep up
     * when a receiver suddenly wants to read a lot of data as it tries to
     * catch up with a deadline.
     */
    if (bp->flags & VSTREAM_FLAG_DEADLINE) {
	timeout = stream->time_limit.tv_sec + (stream->time_limit.tv_usec > 0);
	if (timeout <= 0) {
	    bp->flags |= (VSTREAM_FLAG_RD_ERR | VSTREAM_FLAG_RD_TIMEOUT);
	    errno = ETIMEDOUT;
	    return (VSTREAM_EOF);
	}
	GETTIMEOFDAY(&before);
    } else
	timeout = stream->timeout;
    switch (n = stream->read_fn(stream->fd, bp->data, bp->len, timeout, stream->context)) {
    case -1:
	bp->flags |= VSTREAM_FLAG_RD_ERR;
	if (errno == ETIMEDOUT) {
	    bp->flags |= VSTREAM_FLAG_RD_TIMEOUT;
	    stream->time_limit.tv_sec = stream->time_limit.tv_usec = 0;
	}
	return (VSTREAM_EOF);
    case 0:
	bp->flags |= VSTREAM_FLAG_EOF;
	return (VSTREAM_EOF);
    default:
	if (timeout) {
	    GETTIMEOFDAY(&stream->iotime);
	    if (bp->flags & VSTREAM_FLAG_DEADLINE) {
		VSTREAM_SUB_TIME(elapsed, stream->iotime, before);
		VSTREAM_SUB_TIME(stream->time_limit, stream->time_limit, elapsed);
	    }
	}
	if (msg_verbose > 2)
	    msg_info("%s: fd %d got %ld", myname, stream->fd, (long) n);
	bp->cnt = -n;
	bp->ptr = bp->data;
	if (bp->flags & VSTREAM_FLAG_SEEK)
	    stream->offset += n;
	return (0);
    }
}

/* vstream_buf_put_ready - vbuf callback to make buffer ready for writing */

static int vstream_buf_put_ready(VBUF *bp)
{
    VSTREAM *stream = VBUF_TO_APPL(bp, VSTREAM, buf);
    const char *myname = "vstream_buf_put_ready";

    /*
     * Sanity checks. Detect a change of I/O direction or position. If so,
     * discard unread input, and reset the buffer to the beginning.
     */
    switch (bp->flags & (VSTREAM_FLAG_WRITE | VSTREAM_FLAG_READ)) {
    case VSTREAM_FLAG_READ:			/* change direction */
	bp->flags &= ~VSTREAM_FLAG_READ;
	if (bp->flags & VSTREAM_FLAG_DOUBLE)
	    VSTREAM_SAVE_STATE(stream, read_buf, read_fd);
	/* FALLTHROUGH */
    case 0:					/* change position */
	bp->flags |= VSTREAM_FLAG_WRITE;
	if (bp->flags & VSTREAM_FLAG_DOUBLE)
	    VSTREAM_RESTORE_STATE(stream, write_buf, write_fd);
	else
	    VSTREAM_BUF_AT_START(bp);
	/* FALLTHROUGH */
    case VSTREAM_FLAG_WRITE:			/* no change */
	break;
    case VSTREAM_FLAG_WRITE | VSTREAM_FLAG_READ:
	msg_panic("%s: read/write stream", myname);
    }

    /*
     * Remember the direction. If this is the first PUT operation for this
     * stream or if the buffer is smaller than the requested size, allocate a
     * new buffer; obviously there is no data to be flushed yet. Otherwise,
     * flush the buffer.
     * 
     * XXX Subtle code to set the preferred buffer size as late as possible.
     */
    if (stream->req_bufsize == 0)
	stream->req_bufsize = VSTREAM_BUFSIZE;
    if (bp->len < stream->req_bufsize) {
	vstream_buf_alloc(bp, stream->req_bufsize);
    } else if (bp->cnt <= 0) {
	if (VSTREAM_FFLUSH_SOME(stream))
	    return (VSTREAM_EOF);
    }
    return (0);
}

/* vstream_buf_space - reserve space ahead of time */

static int vstream_buf_space(VBUF *bp, ssize_t want)
{
    VSTREAM *stream = VBUF_TO_APPL(bp, VSTREAM, buf);
    ssize_t used;
    ssize_t incr;
    ssize_t shortage;
    const char *myname = "vstream_buf_space";

    /*
     * Sanity checks. Reserving space implies writing. It is illegal to write
     * to a read-only stream. Detect a change of I/O direction or position.
     * If so, reset the buffer to the beginning.
     */
    if (bp->put_ready == 0)
	msg_panic("%s: read-only stream", myname);
    switch (bp->flags & (VSTREAM_FLAG_READ | VSTREAM_FLAG_WRITE)) {
    case VSTREAM_FLAG_READ:			/* change direction */
	bp->flags &= ~VSTREAM_FLAG_READ;
	if (bp->flags & VSTREAM_FLAG_DOUBLE)
	    VSTREAM_SAVE_STATE(stream, read_buf, read_fd);
	/* FALLTHROUGH */
    case 0:					/* change position */
	bp->flags |= VSTREAM_FLAG_WRITE;
	if (bp->flags & VSTREAM_FLAG_DOUBLE)
	    VSTREAM_RESTORE_STATE(stream, write_buf, write_fd);
	else
	    VSTREAM_BUF_AT_START(bp);
	/* FALLTHROUGH */
    case VSTREAM_FLAG_WRITE:			/* no change */
	break;
    case VSTREAM_FLAG_READ | VSTREAM_FLAG_WRITE:
	msg_panic("%s: read/write stream", myname);
    }

    /*
     * See if enough space is available. If not, flush a multiple of
     * VSTREAM_BUFSIZE bytes and resize the buffer to a multiple of
     * VSTREAM_BUFSIZE. We flush multiples of VSTREAM_BUFSIZE in an attempt
     * to keep file updates block-aligned for better performance.
     * 
     * XXX Subtle code to set the preferred buffer size as late as possible.
     */
#define VSTREAM_TRUNCATE(count, base)	(((count) / (base)) * (base))
#define VSTREAM_ROUNDUP(count, base)	VSTREAM_TRUNCATE(count + base - 1, base)

    if (stream->req_bufsize == 0)
	stream->req_bufsize = VSTREAM_BUFSIZE;
    if (want > bp->cnt) {
	if ((used = bp->len - bp->cnt) > stream->req_bufsize)
	    if (vstream_fflush_some(stream, VSTREAM_TRUNCATE(used, stream->req_bufsize)))
		return (VSTREAM_EOF);
	if ((shortage = (want - bp->cnt)) > 0) {
	    if ((bp->flags & VSTREAM_FLAG_FIXED)
		|| shortage > __MAXINT__(ssize_t) -bp->len - stream->req_bufsize) {
		bp->flags |= VSTREAM_FLAG_WR_ERR;
	    } else {
		incr = VSTREAM_ROUNDUP(shortage, stream->req_bufsize);
		vstream_buf_alloc(bp, bp->len + incr);
	    }
	}
    }
    return (vstream_ferror(stream) ? VSTREAM_EOF : 0);	/* mmap() may fail */
}

/* vstream_fpurge - discard unread or unwritten content */

int     vstream_fpurge(VSTREAM *stream, int direction)
{
    const char *myname = "vstream_fpurge";
    VBUF   *bp = &stream->buf;

#define VSTREAM_MAYBE_PURGE_WRITE(d, b) if ((d) & VSTREAM_PURGE_WRITE) \
	VSTREAM_BUF_AT_START((b))
#define VSTREAM_MAYBE_PURGE_READ(d, b) if ((d) & VSTREAM_PURGE_READ) \
	VSTREAM_BUF_AT_END((b))

    /*
     * To discard all unread contents, position the read buffer at its end,
     * so that we skip over any unread data, and so that the next read
     * operation will refill the buffer.
     * 
     * To discard all unwritten content, position the write buffer at its
     * beginning, so that the next write operation clobbers any unwritten
     * data.
     */
    switch (bp->flags & (VSTREAM_FLAG_READ_DOUBLE | VSTREAM_FLAG_WRITE)) {
    case VSTREAM_FLAG_READ_DOUBLE:
	VSTREAM_MAYBE_PURGE_WRITE(direction, &stream->write_buf);
	/* FALLTHROUGH */
    case VSTREAM_FLAG_READ:
	VSTREAM_MAYBE_PURGE_READ(direction, bp);
	break;
    case VSTREAM_FLAG_DOUBLE:
	VSTREAM_MAYBE_PURGE_WRITE(direction, &stream->write_buf);
	VSTREAM_MAYBE_PURGE_READ(direction, &stream->read_buf);
	break;
    case VSTREAM_FLAG_WRITE_DOUBLE:
	VSTREAM_MAYBE_PURGE_READ(direction, &stream->read_buf);
	/* FALLTHROUGH */
    case VSTREAM_FLAG_WRITE:
	VSTREAM_MAYBE_PURGE_WRITE(direction, bp);
	break;
    case VSTREAM_FLAG_READ_DOUBLE | VSTREAM_FLAG_WRITE:
    case VSTREAM_FLAG_READ | VSTREAM_FLAG_WRITE:
	msg_panic("%s: read/write stream", myname);
    }

    /*
     * Invalidate the cached file seek position.
     */
    bp->flags &= ~VSTREAM_FLAG_SEEK;
    stream->offset = 0;

    return (0);
}

/* vstream_fseek - change I/O position */

off_t   vstream_fseek(VSTREAM *stream, off_t offset, int whence)
{
    const char *myname = "vstream_fseek";
    VBUF   *bp = &stream->buf;

    /*
     * Flush any unwritten output. Discard any unread input. Position the
     * buffer at the end, so that the next GET or PUT operation triggers a
     * buffer boundary action.
     */
    switch (bp->flags & (VSTREAM_FLAG_READ | VSTREAM_FLAG_WRITE)) {
    case VSTREAM_FLAG_WRITE:
	if (bp->ptr > bp->data) {
	    if (whence == SEEK_CUR)
		offset += (bp->ptr - bp->data);	/* add unwritten data */
	    else if (whence == SEEK_END)
		bp->flags &= ~VSTREAM_FLAG_SEEK;
	    if (VSTREAM_FFLUSH_SOME(stream))
		return (-1);
	}
	VSTREAM_BUF_AT_END(bp);
	break;
    case VSTREAM_FLAG_READ:
	if (whence == SEEK_CUR)
	    offset += bp->cnt;			/* subtract unread data */
	else if (whence == SEEK_END)
	    bp->flags &= ~VSTREAM_FLAG_SEEK;
	/* FALLTHROUGH */
    case 0:
	VSTREAM_BUF_AT_END(bp);
	break;
    case VSTREAM_FLAG_READ | VSTREAM_FLAG_WRITE:
	msg_panic("%s: read/write stream", myname);
    }

    /*
     * Clear the read/write flags to inform the buffer boundary action
     * routines that we may have changed I/O position.
     */
    bp->flags &= ~(VSTREAM_FLAG_READ | VSTREAM_FLAG_WRITE);

    /*
     * Shave an unnecessary system call.
     */
    if (bp->flags & VSTREAM_FLAG_NSEEK) {
	errno = ESPIPE;
	return (-1);
    }

    /*
     * Update the cached file seek position.
     */
    if ((stream->offset = lseek(stream->fd, offset, whence)) < 0) {
	if (errno == ESPIPE)
	    bp->flags |= VSTREAM_FLAG_NSEEK;
    } else {
	bp->flags |= VSTREAM_FLAG_SEEK;
    }
    bp->flags &= ~VSTREAM_FLAG_EOF;
    return (stream->offset);
}

/* vstream_ftell - return file offset */

off_t   vstream_ftell(VSTREAM *stream)
{
    VBUF   *bp = &stream->buf;

    /*
     * Shave an unnecessary syscall.
     */
    if (bp->flags & VSTREAM_FLAG_NSEEK) {
	errno = ESPIPE;
	return (-1);
    }

    /*
     * Use the cached file offset when available. This is the offset after
     * the last read, write or seek operation.
     */
    if ((bp->flags & VSTREAM_FLAG_SEEK) == 0) {
	if ((stream->offset = lseek(stream->fd, (off_t) 0, SEEK_CUR)) < 0) {
	    bp->flags |= VSTREAM_FLAG_NSEEK;
	    return (-1);
	}
	bp->flags |= VSTREAM_FLAG_SEEK;
    }

    /*
     * If this is a read buffer, subtract the number of unread bytes from the
     * cached offset. Remember that read counts are negative.
     */
    if (bp->flags & VSTREAM_FLAG_READ)
	return (stream->offset + bp->cnt);

    /*
     * If this is a write buffer, add the number of unwritten bytes to the
     * cached offset.
     */
    if (bp->flags & VSTREAM_FLAG_WRITE)
	return (stream->offset + (bp->ptr - bp->data));

    /*
     * Apparently, this is a new buffer, or a buffer after seek, so there is
     * no need to account for unread or unwritten data.
     */
    return (stream->offset);
}

/* vstream_fdopen - add buffering to pre-opened stream */

VSTREAM *vstream_fdopen(int fd, int flags)
{
    VSTREAM *stream;

    /*
     * Sanity check.
     */
    if (fd < 0)
	msg_panic("vstream_fdopen: bad file %d", fd);

    /*
     * Initialize buffers etc. but do as little as possible. Late buffer
     * allocation etc. gives the application a chance to override default
     * policies. Either this, or the vstream*open() routines would have to
     * have a really ugly interface with lots of mostly-unused arguments (can
     * you say VMS?).
     */
    stream = (VSTREAM *) mymalloc(sizeof(*stream));
    stream->fd = fd;
    stream->read_fn = VSTREAM_CAN_READ(flags) ? (VSTREAM_RW_FN) timed_read : 0;
    stream->write_fn = VSTREAM_CAN_WRITE(flags) ? (VSTREAM_RW_FN) timed_write : 0;
    vstream_buf_init(&stream->buf, flags);
    stream->offset = 0;
    stream->path = 0;
    stream->pid = 0;
    stream->waitpid_fn = 0;
    stream->timeout = 0;
    stream->context = 0;
    stream->jbuf = 0;
    stream->iotime.tv_sec = stream->iotime.tv_usec = 0;
    stream->time_limit.tv_sec = stream->time_limit.tv_usec = 0;
    stream->req_bufsize = 0;
    return (stream);
}

/* vstream_fopen - open buffered file stream */

VSTREAM *vstream_fopen(const char *path, int flags, mode_t mode)
{
    VSTREAM *stream;
    int     fd;

    if ((fd = open(path, flags, mode)) < 0) {
	return (0);
    } else {
	stream = vstream_fdopen(fd, flags);
	stream->path = mystrdup(path);
	return (stream);
    }
}

/* vstream_fflush - flush write buffer */

int     vstream_fflush(VSTREAM *stream)
{
    if ((stream->buf.flags & VSTREAM_FLAG_READ_DOUBLE)
	== VSTREAM_FLAG_READ_DOUBLE
	&& stream->write_buf.len > stream->write_buf.cnt)
	vstream_fflush_delayed(stream);
    return (VSTREAM_FFLUSH_SOME(stream));
}

/* vstream_fclose - close buffered stream */

int     vstream_fclose(VSTREAM *stream)
{
    int     err;

    /*
     * NOTE: Negative file descriptors are not part of the external
     * interface. They are for internal use only, in order to support
     * vstream_fdclose() without a lot of code duplication. Applications that
     * rely on negative VSTREAM file descriptors will break without warning.
     */
    if (stream->pid != 0)
	msg_panic("vstream_fclose: stream has process");
    if ((stream->buf.flags & VSTREAM_FLAG_WRITE_DOUBLE) != 0 && stream->fd >= 0)
	vstream_fflush(stream);
    /* Do not remove: vstream_fdclose() depends on this error test. */
    err = vstream_ferror(stream);
    if (stream->buf.flags & VSTREAM_FLAG_DOUBLE) {
	if (stream->read_fd >= 0)
	    err |= close(stream->read_fd);
	if (stream->write_fd != stream->read_fd)
	    if (stream->write_fd >= 0)
		err |= close(stream->write_fd);
	vstream_buf_wipe(&stream->read_buf);
	vstream_buf_wipe(&stream->write_buf);
	stream->buf = stream->read_buf;
    } else {
	if (stream->fd >= 0)
	    err |= close(stream->fd);
	vstream_buf_wipe(&stream->buf);
    }
    if (stream->path)
	myfree(stream->path);
    if (stream->jbuf)
	myfree((void *) stream->jbuf);
    if (!VSTREAM_STATIC(stream))
	myfree((void *) stream);
    return (err ? VSTREAM_EOF : 0);
}

/* vstream_fdclose - close stream, leave file(s) open */

int     vstream_fdclose(VSTREAM *stream)
{

    /*
     * Flush unwritten output, just like vstream_fclose(). Errors are
     * reported by vstream_fclose().
     */
    if ((stream->buf.flags & VSTREAM_FLAG_WRITE_DOUBLE) != 0)
	(void) vstream_fflush(stream);

    /*
     * NOTE: Negative file descriptors are not part of the external
     * interface. They are for internal use only, in order to support
     * vstream_fdclose() without a lot of code duplication. Applications that
     * rely on negative VSTREAM file descriptors will break without warning.
     */
    if (stream->buf.flags & VSTREAM_FLAG_DOUBLE) {
	stream->fd = stream->read_fd = stream->write_fd = -1;
    } else {
	stream->fd = -1;
    }
    return (vstream_fclose(stream));
}

/* vstream_printf - formatted print to stdout */

VSTREAM *vstream_printf(const char *fmt,...)
{
    VSTREAM *stream = VSTREAM_OUT;
    va_list ap;

    va_start(ap, fmt);
    vbuf_print(&stream->buf, fmt, ap);
    va_end(ap);
    return (stream);
}

/* vstream_fprintf - formatted print to buffered stream */

VSTREAM *vstream_fprintf(VSTREAM *stream, const char *fmt,...)
{
    va_list ap;

    va_start(ap, fmt);
    vbuf_print(&stream->buf, fmt, ap);
    va_end(ap);
    return (stream);
}

/* vstream_fputs - write string to stream */

int     vstream_fputs(const char *str, VSTREAM *stream)
{
    int     ch;

    while ((ch = *str++) != 0)
	if (VSTREAM_PUTC(ch, stream) == VSTREAM_EOF)
	    return (VSTREAM_EOF);
    return (0);
}

/* vstream_control - fine control */

void    vstream_control(VSTREAM *stream, int name,...)
{
    const char *myname = "vstream_control";
    va_list ap;
    int     floor;
    int     old_fd;
    ssize_t req_bufsize = 0;
    VSTREAM *stream2;

#define SWAP(type,a,b) do { type temp = (a); (a) = (b); (b) = (temp); } while (0)

    for (va_start(ap, name); name != VSTREAM_CTL_END; name = va_arg(ap, int)) {
	switch (name) {
	case VSTREAM_CTL_READ_FN:
	    stream->read_fn = va_arg(ap, VSTREAM_RW_FN);
	    break;
	case VSTREAM_CTL_WRITE_FN:
	    stream->write_fn = va_arg(ap, VSTREAM_RW_FN);
	    break;
	case VSTREAM_CTL_CONTEXT:
	    stream->context = va_arg(ap, void *);
	    break;
	case VSTREAM_CTL_PATH:
	    if (stream->path)
		myfree(stream->path);
	    stream->path = mystrdup(va_arg(ap, char *));
	    break;
	case VSTREAM_CTL_DOUBLE:
	    if ((stream->buf.flags & VSTREAM_FLAG_DOUBLE) == 0) {
		stream->buf.flags |= VSTREAM_FLAG_DOUBLE;
		if (stream->buf.flags & VSTREAM_FLAG_READ) {
		    VSTREAM_SAVE_STATE(stream, read_buf, read_fd);
		    VSTREAM_FORK_STATE(stream, write_buf, write_fd);
		} else {
		    VSTREAM_SAVE_STATE(stream, write_buf, write_fd);
		    VSTREAM_FORK_STATE(stream, read_buf, read_fd);
		}
	    }
	    break;
	case VSTREAM_CTL_READ_FD:
	    if ((stream->buf.flags & VSTREAM_FLAG_DOUBLE) == 0)
		msg_panic("VSTREAM_CTL_READ_FD requires double buffering");
	    stream->read_fd = va_arg(ap, int);
	    stream->buf.flags |= VSTREAM_FLAG_NSEEK;
	    break;
	case VSTREAM_CTL_WRITE_FD:
	    if ((stream->buf.flags & VSTREAM_FLAG_DOUBLE) == 0)
		msg_panic("VSTREAM_CTL_WRITE_FD requires double buffering");
	    stream->write_fd = va_arg(ap, int);
	    stream->buf.flags |= VSTREAM_FLAG_NSEEK;
	    break;
	case VSTREAM_CTL_SWAP_FD:
	    stream2 = va_arg(ap, VSTREAM *);
	    if ((stream->buf.flags & VSTREAM_FLAG_DOUBLE)
		!= (stream2->buf.flags & VSTREAM_FLAG_DOUBLE))
		msg_panic("VSTREAM_CTL_SWAP_FD can't swap descriptors between "
			  "single-buffered and double-buffered streams");
	    if (stream->buf.flags & VSTREAM_FLAG_DOUBLE) {
		SWAP(int, stream->read_fd, stream2->read_fd);
		SWAP(int, stream->write_fd, stream2->write_fd);
		stream->fd = ((stream->buf.flags & VSTREAM_FLAG_WRITE) ?
			      stream->write_fd : stream->read_fd);
	    } else {
		SWAP(int, stream->fd, stream2->fd);
	    }
	    break;
	case VSTREAM_CTL_TIMEOUT:
	    if (stream->timeout == 0)
		GETTIMEOFDAY(&stream->iotime);
	    stream->timeout = va_arg(ap, int);
	    if (stream->timeout < 0)
		msg_panic("%s: bad timeout %d", myname, stream->timeout);
	    break;
	case VSTREAM_CTL_EXCEPT:
	    if (stream->jbuf == 0)
		stream->jbuf =
		    (VSTREAM_JMP_BUF *) mymalloc(sizeof(VSTREAM_JMP_BUF));
	    break;

#ifdef VSTREAM_CTL_DUPFD

#define VSTREAM_TRY_DUPFD(backup, fd, floor) do { \
	if (((backup) = (fd)) < floor) { \
	    if (((fd) = fcntl((backup), F_DUPFD, (floor))) < 0) \
		msg_fatal("fcntl F_DUPFD %d: %m", (floor)); \
	    (void) close(backup); \
	} \
    } while (0)

	case VSTREAM_CTL_DUPFD:
	    floor = va_arg(ap, int);
	    if (stream->buf.flags & VSTREAM_FLAG_DOUBLE) {
		VSTREAM_TRY_DUPFD(old_fd, stream->read_fd, floor);
		if (stream->write_fd == old_fd)
		    stream->write_fd = stream->read_fd;
		else
		    VSTREAM_TRY_DUPFD(old_fd, stream->write_fd, floor);
		stream->fd = (stream->buf.flags & VSTREAM_FLAG_READ) ?
		    stream->read_fd : stream->write_fd;
	    } else {
		VSTREAM_TRY_DUPFD(old_fd, stream->fd, floor);
	    }
	    break;
#endif

	    /*
	     * Postpone memory (re)allocation until the space is needed.
	     */
	case VSTREAM_CTL_BUFSIZE:
	    req_bufsize = va_arg(ap, ssize_t);
	    /* Heuristic to detect missing (ssize_t) type cast on LP64 hosts. */
	    if (req_bufsize < 0 || req_bufsize > INT_MAX)
		msg_panic("unreasonable VSTREAM_CTL_BUFSIZE request: %ld",
			  (long) req_bufsize);
	    if ((stream->buf.flags & VSTREAM_FLAG_FIXED) == 0
		&& req_bufsize > stream->req_bufsize) {
		if (msg_verbose)
		    msg_info("fd=%d: stream buffer size old=%ld new=%ld",
			     vstream_fileno(stream),
			     (long) stream->req_bufsize,
			     (long) req_bufsize);
		stream->req_bufsize = req_bufsize;
	    }
	    break;

	    /*
	     * Make no gettimeofday() etc. system call until we really know
	     * that we need to do I/O. This avoids a performance hit when
	     * sending or receiving body content one line at a time.
	     */
	case VSTREAM_CTL_STOP_DEADLINE:
	    stream->buf.flags &= ~VSTREAM_FLAG_DEADLINE;
	    break;
	case VSTREAM_CTL_START_DEADLINE:
	    if (stream->timeout <= 0)
		msg_panic("%s: bad timeout %d", myname, stream->timeout);
	    stream->buf.flags |= VSTREAM_FLAG_DEADLINE;
	    stream->time_limit.tv_sec = stream->timeout;
	    stream->time_limit.tv_usec = 0;
	    break;
	default:
	    msg_panic("%s: bad name %d", myname, name);
	}
    }
    va_end(ap);
}

/* vstream_vprintf - formatted print to stdout */

VSTREAM *vstream_vprintf(const char *format, va_list ap)
{
    VSTREAM *vp = VSTREAM_OUT;

    vbuf_print(&vp->buf, format, ap);
    return (vp);
}

/* vstream_vfprintf - formatted print engine */

VSTREAM *vstream_vfprintf(VSTREAM *vp, const char *format, va_list ap)
{
    vbuf_print(&vp->buf, format, ap);
    return (vp);
}

/* vstream_bufstat - get stream buffer status */

ssize_t vstream_bufstat(VSTREAM *vp, int command)
{
    VBUF   *bp;

    switch (command & VSTREAM_BST_MASK_DIR) {
    case VSTREAM_BST_FLAG_IN:
	if (vp->buf.flags & VSTREAM_FLAG_READ) {
	    bp = &vp->buf;
	} else if (vp->buf.flags & VSTREAM_FLAG_DOUBLE) {
	    bp = &vp->read_buf;
	} else {
	    bp = 0;
	}
	switch (command & ~VSTREAM_BST_MASK_DIR) {
	case VSTREAM_BST_FLAG_PEND:
	    return (bp ? -bp->cnt : 0);
	    /* Add other requests below. */
	}
	break;
    case VSTREAM_BST_FLAG_OUT:
	if (vp->buf.flags & VSTREAM_FLAG_WRITE) {
	    bp = &vp->buf;
	} else if (vp->buf.flags & VSTREAM_FLAG_DOUBLE) {
	    bp = &vp->write_buf;
	} else {
	    bp = 0;
	}
	switch (command & ~VSTREAM_BST_MASK_DIR) {
	case VSTREAM_BST_FLAG_PEND:
	    return (bp ? bp->len - bp->cnt : 0);
	    /* Add other requests below. */
	}
	break;
    }
    msg_panic("vstream_bufstat: unknown command: %d", command);
}

#undef vstream_peek			/* API binary compatibility. */

/* vstream_peek - peek at a stream */

ssize_t vstream_peek(VSTREAM *vp)
{
    if (vp->buf.flags & VSTREAM_FLAG_READ) {
	return (-vp->buf.cnt);
    } else if (vp->buf.flags & VSTREAM_FLAG_DOUBLE) {
	return (-vp->read_buf.cnt);
    } else {
	return (0);
    }
}

/* vstream_peek_data - peek at unread data */

const char *vstream_peek_data(VSTREAM *vp)
{
    if (vp->buf.flags & VSTREAM_FLAG_READ) {
	return ((const char *) vp->buf.ptr);
    } else if (vp->buf.flags & VSTREAM_FLAG_DOUBLE) {
	return ((const char *) vp->read_buf.ptr);
    } else {
	return (0);
    }
}

#ifdef TEST

static void copy_line(ssize_t bufsize)
{
    int     c;

    vstream_control(VSTREAM_IN, CA_VSTREAM_CTL_BUFSIZE(bufsize), VSTREAM_CTL_END);
    vstream_control(VSTREAM_OUT, CA_VSTREAM_CTL_BUFSIZE(bufsize), VSTREAM_CTL_END);
    while ((c = VSTREAM_GETC(VSTREAM_IN)) != VSTREAM_EOF) {
	VSTREAM_PUTC(c, VSTREAM_OUT);
	if (c == '\n')
	    break;
    }
    vstream_fflush(VSTREAM_OUT);
}

static void printf_number(void)
{
    vstream_printf("%d\n", __MAXINT__(int));
    vstream_fflush(VSTREAM_OUT);
}

 /*
  * Exercise some of the features.
  */
int     main(int argc, char **argv)
{

    /*
     * Test buffer expansion and shrinking. Formatted print may silently
     * expand the write buffer and cause multiple bytes to be written.
     */
    copy_line(1);			/* one-byte read/write */
    copy_line(2);				/* two-byte read/write */
    copy_line(1);				/* two-byte read/write */
    printf_number();				/* multi-byte write */
    exit(0);
}

#endif
