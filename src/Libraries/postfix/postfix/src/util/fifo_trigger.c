/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
#include <fcntl.h>
#include <unistd.h>

/* Utility library. */

#include <msg.h>
#include <iostuff.h>
#include <safe_open.h>
#include <trigger.h>

/* fifo_trigger - wakeup fifo server */

int     fifo_trigger(const char *service, const char *buf, ssize_t len, int timeout)
{
    static VSTRING *why;
    const char *myname = "fifo_trigger";
    VSTREAM *fp;
    int     fd;

    if (why == 0)
	why = vstring_alloc(1);

    /*
     * Write the request to the service fifo. According to POSIX, the open
     * shall always return immediately, and shall return an error when no
     * process is reading from the FIFO.
     * 
     * Use safe_open() so that we don't follow symlinks, and so that we don't
     * open files with multiple hard links. We're not (yet) going to bother
     * the caller with safe_open() specific quirks such as the why argument.
     */
    if ((fp = safe_open(service, O_WRONLY | O_NONBLOCK, 0,
			(struct stat *) 0, -1, -1, why)) == 0) {
	if (msg_verbose)
	    msg_info("%s: open %s: %s", myname, service, vstring_str(why));
	return (-1);
    }
    fd = vstream_fileno(fp);

    /*
     * Write the request...
     */
    non_blocking(fd, timeout > 0 ? NON_BLOCKING : BLOCKING);
    if (write_buf(fd, buf, len, timeout) < 0)
	if (msg_verbose)
	    msg_warn("%s: write %s: %m", myname, service);

    /*
     * Disconnect.
     */
    if (vstream_fclose(fp))
	if (msg_verbose)
	    msg_warn("%s: close %s: %m", myname, service);
    return (0);
}

#ifdef TEST

 /*
  * Set up a FIFO listener, and keep triggering until the listener becomes
  * idle, which should never happen.
  */
#include <signal.h>
#include <stdlib.h>

#include "events.h"
#include "listen.h"

#define TEST_FIFO	"test-fifo"

int     trig_count;
int     wakeup_count;

static void cleanup(void)
{
    unlink(TEST_FIFO);
    exit(1);
}

static void handler(int sig)
{
    msg_fatal("got signal %d after %d triggers %d wakeups",
	      sig, trig_count, wakeup_count);
}

static void read_event(int unused_event, char *context)
{
    int     fd = CAST_ANY_PTR_TO_INT(context);
    char    ch;

    wakeup_count++;

    if (read(fd, &ch, 1) != 1)
	msg_fatal("read %s: %m", TEST_FIFO);
}

int     main(int unused_argc, char **unused_argv)
{
    int     listen_fd;

    listen_fd = fifo_listen(TEST_FIFO, 0600, NON_BLOCKING);
    msg_cleanup(cleanup);
    event_enable_read(listen_fd, read_event, CAST_INT_TO_VOID_PTR(listen_fd));
    signal(SIGINT, handler);
    signal(SIGALRM, handler);
    for (;;) {
	alarm(10);
	if (fifo_trigger(TEST_FIFO, "", 1, 0) < 0)
	    msg_fatal("trigger %s: %m", TEST_FIFO);
	trig_count++;
	if (fifo_trigger(TEST_FIFO, "", 1, 0) < 0)
	    msg_fatal("trigger %s: %m", TEST_FIFO);
	trig_count++;
	if (fifo_trigger(TEST_FIFO, "", 1, 0) < 0)
	    msg_fatal("trigger %s: %m", TEST_FIFO);
	trig_count++;
	event_loop(-1);
	event_loop(-1);
	event_loop(-1);
    }
}

#endif
