/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <vstream.h>
#include <events.h>
#include <iostuff.h>
#include <connect.h>
#include <split_at.h>
#include <auto_clnt.h>

/* Application-specific. */

 /*
  * AUTO_CLNT is an opaque structure. None of the access methods can easily
  * be implemented as a macro, and access is not performance critical anyway.
  */
struct AUTO_CLNT {
    VSTREAM *vstream;			/* buffered I/O */
    char   *endpoint;			/* host:port or pathname */
    int     timeout;			/* I/O time limit */
    int     max_idle;			/* time before client disconnect */
    int     max_ttl;			/* time before client disconnect */
    int     (*connect) (const char *, int, int);	/* unix, local, inet */
};

static void auto_clnt_close(AUTO_CLNT *);

/* auto_clnt_event - server-initiated disconnect or client-side max_idle */

static void auto_clnt_event(int unused_event, void *context)
{
    AUTO_CLNT *auto_clnt = (AUTO_CLNT *) context;

    /*
     * Sanity check. This routine causes the stream to be closed, so it
     * cannot be called when the stream is already closed.
     */
    if (auto_clnt->vstream == 0)
	msg_panic("auto_clnt_event: stream is closed");

    auto_clnt_close(auto_clnt);
}

/* auto_clnt_ttl_event - client-side expiration */

static void auto_clnt_ttl_event(int event, void *context)
{

    /*
     * XXX This function is needed only because event_request_timer() cannot
     * distinguish between requests that specify the same call-back routine
     * and call-back context. The fix is obvious: specify a request ID along
     * with the call-back routine, but there is too much code that would have
     * to be changed.
     * 
     * XXX Should we be concerned that an overly aggressive optimizer will
     * eliminate this function and replace calls to auto_clnt_ttl_event() by
     * direct calls to auto_clnt_event()? It should not, because there exists
     * code that takes the address of both functions.
     */
    auto_clnt_event(event, context);
}

/* auto_clnt_open - connect to service */

static void auto_clnt_open(AUTO_CLNT *auto_clnt)
{
    const char *myname = "auto_clnt_open";
    int     fd;

    /*
     * Sanity check.
     */
    if (auto_clnt->vstream)
	msg_panic("auto_clnt_open: stream is open");

    /*
     * Schedule a read event so that we can clean up when the remote side
     * disconnects, and schedule a timer event so that we can cleanup an idle
     * connection. Note that both events are handled by the same routine.
     * 
     * Finally, schedule an event to force disconnection even when the
     * connection is not idle. This is to prevent one client from clinging on
     * to a server forever.
     */
    fd = auto_clnt->connect(auto_clnt->endpoint, BLOCKING, auto_clnt->timeout);
    if (fd < 0) {
	msg_warn("connect to %s: %m", auto_clnt->endpoint);
    } else {
	if (msg_verbose)
	    msg_info("%s: connected to %s", myname, auto_clnt->endpoint);
	auto_clnt->vstream = vstream_fdopen(fd, O_RDWR);
	vstream_control(auto_clnt->vstream,
			CA_VSTREAM_CTL_PATH(auto_clnt->endpoint),
			CA_VSTREAM_CTL_TIMEOUT(auto_clnt->timeout),
			CA_VSTREAM_CTL_END);
    }

    if (auto_clnt->vstream != 0) {
	close_on_exec(vstream_fileno(auto_clnt->vstream), CLOSE_ON_EXEC);
	event_enable_read(vstream_fileno(auto_clnt->vstream), auto_clnt_event,
			  (void *) auto_clnt);
	if (auto_clnt->max_idle > 0)
	    event_request_timer(auto_clnt_event, (void *) auto_clnt,
				auto_clnt->max_idle);
	if (auto_clnt->max_ttl > 0)
	    event_request_timer(auto_clnt_ttl_event, (void *) auto_clnt,
				auto_clnt->max_ttl);
    }
}

/* auto_clnt_close - disconnect from service */

static void auto_clnt_close(AUTO_CLNT *auto_clnt)
{
    const char *myname = "auto_clnt_close";

    /*
     * Sanity check.
     */
    if (auto_clnt->vstream == 0)
	msg_panic("%s: stream is closed", myname);

    /*
     * Be sure to disable read and timer events.
     */
    if (msg_verbose)
	msg_info("%s: disconnect %s stream",
		 myname, VSTREAM_PATH(auto_clnt->vstream));
    event_disable_readwrite(vstream_fileno(auto_clnt->vstream));
    event_cancel_timer(auto_clnt_event, (void *) auto_clnt);
    event_cancel_timer(auto_clnt_ttl_event, (void *) auto_clnt);
    (void) vstream_fclose(auto_clnt->vstream);
    auto_clnt->vstream = 0;
}

/* auto_clnt_recover - recover from server-initiated disconnect */

void    auto_clnt_recover(AUTO_CLNT *auto_clnt)
{

    /*
     * Clean up. Don't re-connect until the caller needs it.
     */
    if (auto_clnt->vstream)
	auto_clnt_close(auto_clnt);
}

/* auto_clnt_access - access a client stream */

VSTREAM *auto_clnt_access(AUTO_CLNT *auto_clnt)
{

    /*
     * Open a stream or restart the idle timer.
     * 
     * Important! Do not restart the TTL timer!
     */
    if (auto_clnt->vstream == 0) {
	auto_clnt_open(auto_clnt);
    } else {
	if (auto_clnt->max_idle > 0)
	    event_request_timer(auto_clnt_event, (void *) auto_clnt,
				auto_clnt->max_idle);
    }
    return (auto_clnt->vstream);
}

/* auto_clnt_create - create client stream object */

AUTO_CLNT *auto_clnt_create(const char *service, int timeout,
			            int max_idle, int max_ttl)
{
    const char *myname = "auto_clnt_create";
    char   *transport = mystrdup(service);
    char   *endpoint;
    AUTO_CLNT *auto_clnt;

    /*
     * Don't open the stream until the caller needs it.
     */
    if ((endpoint = split_at(transport, ':')) == 0
	|| *endpoint == 0 || *transport == 0)
	msg_fatal("need service transport:endpoint instead of \"%s\"", service);
    if (msg_verbose)
	msg_info("%s: transport=%s endpoint=%s", myname, transport, endpoint);
    auto_clnt = (AUTO_CLNT *) mymalloc(sizeof(*auto_clnt));
    auto_clnt->vstream = 0;
    auto_clnt->endpoint = mystrdup(endpoint);
    auto_clnt->timeout = timeout;
    auto_clnt->max_idle = max_idle;
    auto_clnt->max_ttl = max_ttl;
    if (strcmp(transport, "inet") == 0) {
	auto_clnt->connect = inet_connect;
    } else if (strcmp(transport, "local") == 0) {
	auto_clnt->connect = LOCAL_CONNECT;
    } else if (strcmp(transport, "unix") == 0) {
	auto_clnt->connect = unix_connect;
    } else {
	msg_fatal("invalid transport name: %s in service: %s",
		  transport, service);
    }
    myfree(transport);
    return (auto_clnt);
}

/* auto_clnt_name - return client stream name */

const char *auto_clnt_name(AUTO_CLNT *auto_clnt)
{
    return (auto_clnt->endpoint);
}

/* auto_clnt_free - destroy client stream instance */

void    auto_clnt_free(AUTO_CLNT *auto_clnt)
{
    if (auto_clnt->vstream)
	auto_clnt_close(auto_clnt);
    myfree(auto_clnt->endpoint);
    myfree((void *) auto_clnt);
}
