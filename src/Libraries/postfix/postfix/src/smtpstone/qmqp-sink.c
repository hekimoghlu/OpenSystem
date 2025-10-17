/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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
#include <sys/wait.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <signal.h>

/* Utility library. */

#include <msg.h>
#include <vstring.h>
#include <vstream.h>
#include <listen.h>
#include <events.h>
#include <mymalloc.h>
#include <iostuff.h>
#include <msg_vstream.h>
#include <netstring.h>
#include <inet_proto.h>

/* Global library. */

#include <qmqp_proto.h>
#include <mail_version.h>

/* Application-specific. */

typedef struct {
    VSTREAM *stream;			/* client connection */
    int     count;			/* bytes to go */
} SINK_STATE;

static int var_tmout;
static VSTRING *buffer;
static void disconnect(SINK_STATE *);
static int count_deliveries;
static int counter;

/* send_reply - finish conversation */

static void send_reply(SINK_STATE *state)
{
    vstring_sprintf(buffer, "%cOk", QMQP_STAT_OK);
    NETSTRING_PUT_BUF(state->stream, buffer);
    netstring_fflush(state->stream);
    if (count_deliveries) {
	counter++;
	vstream_printf("%d\r", counter);
	vstream_fflush(VSTREAM_OUT);
    }
    disconnect(state);
}

/* read_data - read over-all netstring data */

static void read_data(int unused_event, void *context)
{
    SINK_STATE *state = (SINK_STATE *) context;
    int     fd = vstream_fileno(state->stream);
    int     count;

    /*
     * Refill the VSTREAM buffer, if necessary.
     */
    if (VSTREAM_GETC(state->stream) == VSTREAM_EOF)
	netstring_except(state->stream, vstream_ftimeout(state->stream) ?
			 NETSTRING_ERR_TIME : NETSTRING_ERR_EOF);
    state->count--;

    /*
     * Flush the VSTREAM buffer. As documented, vstream_fseek() discards
     * unread input.
     */
    if ((count = vstream_peek(state->stream)) > 0) {
	state->count -= count;
	if (state->count <= 0) {
	    send_reply(state);
	    return;
	}
	vstream_fpurge(state->stream, VSTREAM_PURGE_BOTH);
    }

    /*
     * Do not block while waiting for the arrival of more data.
     */
    event_disable_readwrite(fd);
    event_enable_read(fd, read_data, context);
}

/* read_length - read over-all netstring length */

static void read_length(int event, void *context)
{
    SINK_STATE *state = (SINK_STATE *) context;

    switch (vstream_setjmp(state->stream)) {

    default:
	msg_panic("unknown error reading input");

    case NETSTRING_ERR_TIME:
	msg_panic("attempt to read non-readable socket");
	/* NOTREACHED */

    case NETSTRING_ERR_EOF:
	msg_warn("lost connection");
	disconnect(state);
	return;

    case NETSTRING_ERR_FORMAT:
	msg_warn("netstring format error");
	disconnect(state);
	return;

    case NETSTRING_ERR_SIZE:
	msg_warn("netstring size error");
	disconnect(state);
	return;

	/*
	 * Include the netstring terminator in the read byte count. This
	 * violates abstractions.
	 */
    case 0:
	state->count = netstring_get_length(state->stream) + 1;
	read_data(event, context);
	return;
    }
}

/* disconnect - handle disconnection events */

static void disconnect(SINK_STATE *state)
{
    event_disable_readwrite(vstream_fileno(state->stream));
    vstream_fclose(state->stream);
    myfree((void *) state);
}

/* connect_event - handle connection events */

static void connect_event(int unused_event, void *context)
{
    int     sock = CAST_ANY_PTR_TO_INT(context);
    struct sockaddr_storage ss;
    SOCKADDR_SIZE len = sizeof(ss);
    struct sockaddr *sa = (struct sockaddr *) &ss;
    SINK_STATE *state;
    int     fd;

    if ((fd = accept(sock, sa, &len)) >= 0) {
	if (msg_verbose)
	    msg_info("connect (%s)",
#ifdef AF_LOCAL
		     sa->sa_family == AF_LOCAL ? "AF_LOCAL" :
#else
		     sa->sa_family == AF_UNIX ? "AF_UNIX" :
#endif
		     sa->sa_family == AF_INET ? "AF_INET" :
#ifdef AF_INET6
		     sa->sa_family == AF_INET6 ? "AF_INET6" :
#endif
		     "unknown protocol family");
	non_blocking(fd, NON_BLOCKING);
	state = (SINK_STATE *) mymalloc(sizeof(*state));
	state->stream = vstream_fdopen(fd, O_RDWR);
	vstream_tweak_sock(state->stream);
	netstring_setup(state->stream, var_tmout);
	event_enable_read(fd, read_length, (void *) state);
    }
}

/* terminate - voluntary exit */

static void terminate(int unused_event, void *unused_context)
{
    exit(0);
}

/* usage - explain */

static void usage(char *myname)
{
    msg_fatal("usage: %s [-cv] [-x time] [host]:port backlog", myname);
}

MAIL_VERSION_STAMP_DECLARE;

int     main(int argc, char **argv)
{
    int     sock;
    int     backlog;
    int     ch;
    int     ttl;
    const char *protocols = INET_PROTO_NAME_ALL;

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    /*
     * Fix 20051207.
     */
    signal(SIGPIPE, SIG_IGN);

    /*
     * Initialize diagnostics.
     */
    msg_vstream_init(argv[0], VSTREAM_ERR);

    /*
     * Parse JCL.
     */
    while ((ch = GETOPT(argc, argv, "46cvx:")) > 0) {
	switch (ch) {
	case '4':
	    protocols = INET_PROTO_NAME_IPV4;
	    break;
	case '6':
	    protocols = INET_PROTO_NAME_IPV6;
	    break;
	case 'c':
	    count_deliveries++;
	    break;
	case 'v':
	    msg_verbose++;
	    break;
	case 'x':
	    if ((ttl = atoi(optarg)) <= 0)
		usage(argv[0]);
	    event_request_timer(terminate, (void *) 0, ttl);
	    break;
	default:
	    usage(argv[0]);
	}
    }
    if (argc - optind != 2)
	usage(argv[0]);
    if ((backlog = atoi(argv[optind + 1])) <= 0)
	usage(argv[0]);

    /*
     * Initialize.
     */
    (void) inet_proto_init("protocols", protocols);
    buffer = vstring_alloc(1024);
    if (strncmp(argv[optind], "unix:", 5) == 0) {
	sock = unix_listen(argv[optind] + 5, backlog, BLOCKING);
    } else {
	if (strncmp(argv[optind], "inet:", 5) == 0)
	    argv[optind] += 5;
	sock = inet_listen(argv[optind], backlog, BLOCKING);
    }

    /*
     * Start the event handler.
     */
    event_enable_read(sock, connect_event, CAST_INT_TO_VOID_PTR(sock));
    for (;;)
	event_loop(-1);
}
