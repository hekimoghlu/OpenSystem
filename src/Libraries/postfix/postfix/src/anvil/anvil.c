/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 6, 2023.
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
#include <limits.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <htable.h>
#include <stringops.h>
#include <events.h>

/* Global library. */

#include <mail_conf.h>
#include <mail_params.h>
#include <mail_version.h>
#include <mail_proto.h>
#include <anvil_clnt.h>

/* Server skeleton. */

#include <mail_server.h>

/* Application-specific. */

 /*
  * Configuration parameters.
  */
int     var_anvil_time_unit;
int     var_anvil_stat_time;

 /*
  * Global dynamic state.
  */
static HTABLE *anvil_remote_map;	/* indexed by service+ remote client */

 /*
  * Remote connection state, one instance for each (service, client) pair.
  */
typedef struct {
    char   *ident;			/* lookup key */
    int     count;			/* connection count */
    int     rate;			/* connection rate */
    int     mail;			/* message rate */
    int     rcpt;			/* recipient rate */
    int     ntls;			/* new TLS session rate */
    int     auth;			/* AUTH request rate */
    time_t  start;			/* time of first rate sample */
} ANVIL_REMOTE;

 /*
  * Local server state, one instance per anvil client connection. This allows
  * us to clean up remote connection state when a local server goes away
  * without cleaning up.
  */
typedef struct {
    ANVIL_REMOTE *anvil_remote;		/* XXX should be list */
} ANVIL_LOCAL;

 /*
  * The following operations are implemented as macros with recognizable
  * names so that we don't lose sight of what the code is trying to do.
  * 
  * Related operations are defined side by side so that the code implementing
  * them isn't pages apart.
  */

/* Create new (service, client) state. */

#define ANVIL_REMOTE_FIRST_CONN(remote, id) \
    do { \
	(remote)->ident = mystrdup(id); \
	(remote)->count = 1; \
	(remote)->rate = 1; \
	(remote)->mail = 0; \
	(remote)->rcpt = 0; \
	(remote)->ntls = 0; \
	(remote)->auth = 0; \
	(remote)->start = event_time(); \
    } while(0)

/* Destroy unused (service, client) state. */

#define ANVIL_REMOTE_FREE(remote) \
    do { \
	myfree((remote)->ident); \
	myfree((void *) (remote)); \
    } while(0)

/* Reset or update rate information for existing (service, client) state. */

#define ANVIL_REMOTE_RSET_RATE(remote, _start) \
    do { \
	(remote)->rate = 0; \
	(remote)->mail = 0; \
	(remote)->rcpt = 0; \
	(remote)->ntls = 0; \
	(remote)->auth = 0; \
	(remote)->start = _start; \
    } while(0)

#define ANVIL_REMOTE_INCR_RATE(remote, _what) \
    do { \
	time_t _now = event_time(); \
	if ((remote)->start + var_anvil_time_unit < _now) \
	    ANVIL_REMOTE_RSET_RATE((remote), _now); \
	if ((remote)->_what < INT_MAX) \
            (remote)->_what += 1; \
    } while(0)

/* Update existing (service, client) state. */

#define ANVIL_REMOTE_NEXT_CONN(remote) \
    do { \
	ANVIL_REMOTE_INCR_RATE((remote), rate); \
	if ((remote)->count == 0) \
	    event_cancel_timer(anvil_remote_expire, (void *) remote); \
	(remote)->count++; \
    } while(0)

#define ANVIL_REMOTE_INCR_MAIL(remote)	ANVIL_REMOTE_INCR_RATE((remote), mail)

#define ANVIL_REMOTE_INCR_RCPT(remote)	ANVIL_REMOTE_INCR_RATE((remote), rcpt)

#define ANVIL_REMOTE_INCR_NTLS(remote)	ANVIL_REMOTE_INCR_RATE((remote), ntls)

#define ANVIL_REMOTE_INCR_AUTH(remote)	ANVIL_REMOTE_INCR_RATE((remote), auth)

/* Drop connection from (service, client) state. */

#define ANVIL_REMOTE_DROP_ONE(remote) \
    do { \
	if ((remote) && (remote)->count > 0) { \
	    if (--(remote)->count == 0) \
		event_request_timer(anvil_remote_expire, (void *) remote, \
			var_anvil_time_unit); \
	} \
    } while(0)

/* Create local server state. */

#define ANVIL_LOCAL_INIT(local) \
    do { \
	(local)->anvil_remote = 0; \
    } while(0)

/* Add remote connection to local server. */

#define ANVIL_LOCAL_ADD_ONE(local, remote) \
    do { \
	/* XXX allow multiple remote clients per local server. */ \
	if ((local)->anvil_remote) \
	    ANVIL_REMOTE_DROP_ONE((local)->anvil_remote); \
	(local)->anvil_remote = (remote); \
    } while(0)

/* Test if this remote connection is listed for this local server. */

#define ANVIL_LOCAL_REMOTE_LINKED(local, remote) \
    ((local)->anvil_remote == (remote))

/* Drop specific remote connection from local server. */

#define ANVIL_LOCAL_DROP_ONE(local, remote) \
    do { \
	/* XXX allow multiple remote clients per local server. */ \
	if ((local)->anvil_remote == (remote)) \
	    (local)->anvil_remote = 0; \
    } while(0)

/* Drop all remote connections from local server. */

#define ANVIL_LOCAL_DROP_ALL(stream, local) \
    do { \
	 /* XXX allow multiple remote clients per local server. */ \
	if ((local)->anvil_remote) \
	    anvil_remote_disconnect((stream), (local)->anvil_remote->ident); \
    } while (0)

 /*
  * Lookup table to map request names to action routines.
  */
typedef struct {
    const char *name;
    void    (*action) (VSTREAM *, const char *);
} ANVIL_REQ_TABLE;

 /*
  * Run-time statistics for maximal connection counts and event rates. These
  * store the peak resource usage, remote connection, and time. Absent a
  * query interface, this information is logged at process exit time and at
  * configurable intervals.
  */
typedef struct {
    int     value;			/* peak value */
    char   *ident;			/* lookup key */
    time_t  when;			/* time of peak value */
} ANVIL_MAX;

static ANVIL_MAX max_conn_count;	/* peak connection count */
static ANVIL_MAX max_conn_rate;		/* peak connection rate */
static ANVIL_MAX max_mail_rate;		/* peak message rate */
static ANVIL_MAX max_rcpt_rate;		/* peak recipient rate */
static ANVIL_MAX max_ntls_rate;		/* peak new TLS session rate */
static ANVIL_MAX max_auth_rate;		/* peak AUTH request rate */

static int max_cache_size;		/* peak cache size */
static time_t max_cache_time;		/* time of peak size */

/* Update/report peak usage. */

#define ANVIL_MAX_UPDATE(_max, _value, _ident) \
    do { \
	_max.value = _value; \
	if (_max.ident == 0) { \
	    _max.ident = mystrdup(_ident); \
	} else if (!STREQ(_max.ident, _ident)) { \
	    myfree(_max.ident); \
	    _max.ident = mystrdup(_ident); \
	} \
	_max.when = event_time(); \
    } while (0)

#define ANVIL_MAX_RATE_REPORT(_max, _name) \
    do { \
	if (_max.value > 0) { \
	    msg_info("statistics: max " _name " rate %d/%ds for (%s) at %.15s", \
		_max.value, var_anvil_time_unit, \
		_max.ident, ctime(&_max.when) + 4); \
	    _max.value = 0; \
	} \
    } while (0);

#define ANVIL_MAX_COUNT_REPORT(_max, _name) \
    do { \
	if (_max.value > 0) { \
	    msg_info("statistics: max " _name " count %d for (%s) at %.15s", \
		_max.value, _max.ident, ctime(&_max.when) + 4); \
	    _max.value = 0; \
	} \
    } while (0);

 /*
  * Silly little macros.
  */
#define STR(x)			vstring_str(x)
#define STREQ(x,y)		(strcmp((x), (y)) == 0)

/* anvil_remote_expire - purge expired connection state */

static void anvil_remote_expire(int unused_event, void *context)
{
    ANVIL_REMOTE *anvil_remote = (ANVIL_REMOTE *) context;
    const char *myname = "anvil_remote_expire";

    if (msg_verbose)
	msg_info("%s %s", myname, anvil_remote->ident);

    if (anvil_remote->count != 0)
	msg_panic("%s: bad connection count: %d",
		  myname, anvil_remote->count);

    htable_delete(anvil_remote_map, anvil_remote->ident,
		  (void (*) (void *)) 0);
    ANVIL_REMOTE_FREE(anvil_remote);

    if (msg_verbose)
	msg_info("%s: anvil_remote_map used=%ld",
		 myname, (long) anvil_remote_map->used);
}

/* anvil_remote_lookup - dump address status */

static void anvil_remote_lookup(VSTREAM *client_stream, const char *ident)
{
    ANVIL_REMOTE *anvil_remote;
    const char *myname = "anvil_remote_lookup";

    if (msg_verbose)
	msg_info("%s fd=%d stream=0x%lx ident=%s",
		 myname, vstream_fileno(client_stream),
		 (unsigned long) client_stream, ident);

    /*
     * Look up remote client information.
     */
    if ((anvil_remote =
	 (ANVIL_REMOTE *) htable_find(anvil_remote_map, ident)) == 0) {
	attr_print_plain(client_stream, ATTR_FLAG_NONE,
			 SEND_ATTR_INT(ANVIL_ATTR_STATUS, ANVIL_STAT_OK),
			 SEND_ATTR_INT(ANVIL_ATTR_COUNT, 0),
			 SEND_ATTR_INT(ANVIL_ATTR_RATE, 0),
			 SEND_ATTR_INT(ANVIL_ATTR_MAIL, 0),
			 SEND_ATTR_INT(ANVIL_ATTR_RCPT, 0),
			 SEND_ATTR_INT(ANVIL_ATTR_NTLS, 0),
			 SEND_ATTR_INT(ANVIL_ATTR_AUTH, 0),
			 ATTR_TYPE_END);
    } else {

	/*
	 * Do not report stale information.
	 */
	if (anvil_remote->start != 0
	    && anvil_remote->start + var_anvil_time_unit < event_time())
	    ANVIL_REMOTE_RSET_RATE(anvil_remote, 0);
	attr_print_plain(client_stream, ATTR_FLAG_NONE,
			 SEND_ATTR_INT(ANVIL_ATTR_STATUS, ANVIL_STAT_OK),
		       SEND_ATTR_INT(ANVIL_ATTR_COUNT, anvil_remote->count),
			 SEND_ATTR_INT(ANVIL_ATTR_RATE, anvil_remote->rate),
			 SEND_ATTR_INT(ANVIL_ATTR_MAIL, anvil_remote->mail),
			 SEND_ATTR_INT(ANVIL_ATTR_RCPT, anvil_remote->rcpt),
			 SEND_ATTR_INT(ANVIL_ATTR_NTLS, anvil_remote->ntls),
			 SEND_ATTR_INT(ANVIL_ATTR_AUTH, anvil_remote->auth),
			 ATTR_TYPE_END);
    }
}

/* anvil_remote_conn_update - instantiate or update connection info */

static ANVIL_REMOTE *anvil_remote_conn_update(VSTREAM *client_stream, const char *ident)
{
    ANVIL_REMOTE *anvil_remote;
    ANVIL_LOCAL *anvil_local;
    const char *myname = "anvil_remote_conn_update";

    if (msg_verbose)
	msg_info("%s fd=%d stream=0x%lx ident=%s",
		 myname, vstream_fileno(client_stream),
		 (unsigned long) client_stream, ident);

    /*
     * Look up remote connection count information. Update remote connection
     * rate information. Simply reset the counter every var_anvil_time_unit
     * seconds. This is easier than maintaining a moving average and it gives
     * a quicker response to tresspassers.
     */
    if ((anvil_remote =
	 (ANVIL_REMOTE *) htable_find(anvil_remote_map, ident)) == 0) {
	anvil_remote = (ANVIL_REMOTE *) mymalloc(sizeof(*anvil_remote));
	ANVIL_REMOTE_FIRST_CONN(anvil_remote, ident);
	htable_enter(anvil_remote_map, ident, (void *) anvil_remote);
	if (max_cache_size < anvil_remote_map->used) {
	    max_cache_size = anvil_remote_map->used;
	    max_cache_time = event_time();
	}
    } else {
	ANVIL_REMOTE_NEXT_CONN(anvil_remote);
    }

    /*
     * Record this connection under the local server information, so that we
     * can clean up all its connection state when the local server goes away.
     */
    if ((anvil_local = (ANVIL_LOCAL *) vstream_context(client_stream)) == 0) {
	anvil_local = (ANVIL_LOCAL *) mymalloc(sizeof(*anvil_local));
	ANVIL_LOCAL_INIT(anvil_local);
	vstream_control(client_stream,
			CA_VSTREAM_CTL_CONTEXT((void *) anvil_local),
			CA_VSTREAM_CTL_END);
    }
    ANVIL_LOCAL_ADD_ONE(anvil_local, anvil_remote);
    if (msg_verbose)
	msg_info("%s: anvil_local 0x%lx",
		 myname, (unsigned long) anvil_local);

    return (anvil_remote);
}

/* anvil_remote_connect - report connection event, query address status */

static void anvil_remote_connect(VSTREAM *client_stream, const char *ident)
{
    ANVIL_REMOTE *anvil_remote;

    /*
     * Update or instantiate connection info.
     */
    anvil_remote = anvil_remote_conn_update(client_stream, ident);

    /*
     * Respond to the local server.
     */
    attr_print_plain(client_stream, ATTR_FLAG_NONE,
		     SEND_ATTR_INT(ANVIL_ATTR_STATUS, ANVIL_STAT_OK),
		     SEND_ATTR_INT(ANVIL_ATTR_COUNT, anvil_remote->count),
		     SEND_ATTR_INT(ANVIL_ATTR_RATE, anvil_remote->rate),
		     ATTR_TYPE_END);

    /*
     * Update peak statistics.
     */
    if (anvil_remote->rate > max_conn_rate.value)
	ANVIL_MAX_UPDATE(max_conn_rate, anvil_remote->rate, anvil_remote->ident);
    if (anvil_remote->count > max_conn_count.value)
	ANVIL_MAX_UPDATE(max_conn_count, anvil_remote->count, anvil_remote->ident);
}

/* anvil_remote_mail - register message delivery request */

static void anvil_remote_mail(VSTREAM *client_stream, const char *ident)
{
    ANVIL_REMOTE *anvil_remote;

    /*
     * Be prepared for "postfix reload" after "connect".
     */
    if ((anvil_remote =
	 (ANVIL_REMOTE *) htable_find(anvil_remote_map, ident)) == 0)
	anvil_remote = anvil_remote_conn_update(client_stream, ident);

    /*
     * Update message delivery request rate and respond to local server.
     */
    ANVIL_REMOTE_INCR_MAIL(anvil_remote);
    attr_print_plain(client_stream, ATTR_FLAG_NONE,
		     SEND_ATTR_INT(ANVIL_ATTR_STATUS, ANVIL_STAT_OK),
		     SEND_ATTR_INT(ANVIL_ATTR_RATE, anvil_remote->mail),
		     ATTR_TYPE_END);

    /*
     * Update peak statistics.
     */
    if (anvil_remote->mail > max_mail_rate.value)
	ANVIL_MAX_UPDATE(max_mail_rate, anvil_remote->mail, anvil_remote->ident);
}

/* anvil_remote_rcpt - register recipient address event */

static void anvil_remote_rcpt(VSTREAM *client_stream, const char *ident)
{
    ANVIL_REMOTE *anvil_remote;

    /*
     * Be prepared for "postfix reload" after "connect".
     */
    if ((anvil_remote =
	 (ANVIL_REMOTE *) htable_find(anvil_remote_map, ident)) == 0)
	anvil_remote = anvil_remote_conn_update(client_stream, ident);

    /*
     * Update recipient address rate and respond to local server.
     */
    ANVIL_REMOTE_INCR_RCPT(anvil_remote);
    attr_print_plain(client_stream, ATTR_FLAG_NONE,
		     SEND_ATTR_INT(ANVIL_ATTR_STATUS, ANVIL_STAT_OK),
		     SEND_ATTR_INT(ANVIL_ATTR_RATE, anvil_remote->rcpt),
		     ATTR_TYPE_END);

    /*
     * Update peak statistics.
     */
    if (anvil_remote->rcpt > max_rcpt_rate.value)
	ANVIL_MAX_UPDATE(max_rcpt_rate, anvil_remote->rcpt, anvil_remote->ident);
}

/* anvil_remote_auth - register auth request event */

static void anvil_remote_auth(VSTREAM *client_stream, const char *ident)
{
    ANVIL_REMOTE *anvil_remote;

    /*
     * Be prepared for "postfix reload" after "connect".
     */
    if ((anvil_remote =
	 (ANVIL_REMOTE *) htable_find(anvil_remote_map, ident)) == 0)
	anvil_remote = anvil_remote_conn_update(client_stream, ident);

    /*
     * Update recipient address rate and respond to local server.
     */
    ANVIL_REMOTE_INCR_AUTH(anvil_remote);
    attr_print_plain(client_stream, ATTR_FLAG_NONE,
		     SEND_ATTR_INT(ANVIL_ATTR_STATUS, ANVIL_STAT_OK),
		     SEND_ATTR_INT(ANVIL_ATTR_RATE, anvil_remote->auth),
		     ATTR_TYPE_END);

    /*
     * Update peak statistics.
     */
    if (anvil_remote->auth > max_auth_rate.value)
	ANVIL_MAX_UPDATE(max_auth_rate, anvil_remote->auth, anvil_remote->ident);
}

/* anvil_remote_newtls - register newtls event */

static void anvil_remote_newtls(VSTREAM *client_stream, const char *ident)
{
    ANVIL_REMOTE *anvil_remote;

    /*
     * Be prepared for "postfix reload" after "connect".
     */
    if ((anvil_remote =
	 (ANVIL_REMOTE *) htable_find(anvil_remote_map, ident)) == 0)
	anvil_remote = anvil_remote_conn_update(client_stream, ident);

    /*
     * Update newtls rate and respond to local server.
     */
    ANVIL_REMOTE_INCR_NTLS(anvil_remote);
    attr_print_plain(client_stream, ATTR_FLAG_NONE,
		     SEND_ATTR_INT(ANVIL_ATTR_STATUS, ANVIL_STAT_OK),
		     SEND_ATTR_INT(ANVIL_ATTR_RATE, anvil_remote->ntls),
		     ATTR_TYPE_END);

    /*
     * Update peak statistics.
     */
    if (anvil_remote->ntls > max_ntls_rate.value)
	ANVIL_MAX_UPDATE(max_ntls_rate, anvil_remote->ntls, anvil_remote->ident);
}

/* anvil_remote_newtls_stat - report newtls stats */

static void anvil_remote_newtls_stat(VSTREAM *client_stream, const char *ident)
{
    ANVIL_REMOTE *anvil_remote;
    int     rate;

    /*
     * Be prepared for "postfix reload" after "connect".
     */
    if ((anvil_remote =
	 (ANVIL_REMOTE *) htable_find(anvil_remote_map, ident)) == 0) {
	rate = 0;
    }

    /*
     * Do not report stale information.
     */
    else {
	if (anvil_remote->start != 0
	    && anvil_remote->start + var_anvil_time_unit < event_time())
	    ANVIL_REMOTE_RSET_RATE(anvil_remote, 0);
	rate = anvil_remote->ntls;
    }

    /*
     * Respond to local server.
     */
    attr_print_plain(client_stream, ATTR_FLAG_NONE,
		     SEND_ATTR_INT(ANVIL_ATTR_STATUS, ANVIL_STAT_OK),
		     SEND_ATTR_INT(ANVIL_ATTR_RATE, rate),
		     ATTR_TYPE_END);
}

/* anvil_remote_disconnect - report disconnect event */

static void anvil_remote_disconnect(VSTREAM *client_stream, const char *ident)
{
    ANVIL_REMOTE *anvil_remote;
    ANVIL_LOCAL *anvil_local;
    const char *myname = "anvil_remote_disconnect";

    if (msg_verbose)
	msg_info("%s fd=%d stream=0x%lx ident=%s",
		 myname, vstream_fileno(client_stream),
		 (unsigned long) client_stream, ident);

    /*
     * Update local and remote info if this remote connection is listed for
     * this local server.
     */
    if ((anvil_local = (ANVIL_LOCAL *) vstream_context(client_stream)) != 0
	&& (anvil_remote =
	    (ANVIL_REMOTE *) htable_find(anvil_remote_map, ident)) != 0
	&& ANVIL_LOCAL_REMOTE_LINKED(anvil_local, anvil_remote)) {
	ANVIL_REMOTE_DROP_ONE(anvil_remote);
	ANVIL_LOCAL_DROP_ONE(anvil_local, anvil_remote);
    }
    if (msg_verbose)
	msg_info("%s: anvil_local 0x%lx",
		 myname, (unsigned long) anvil_local);

    /*
     * Respond to the local server.
     */
    attr_print_plain(client_stream, ATTR_FLAG_NONE,
		     SEND_ATTR_INT(ANVIL_ATTR_STATUS, ANVIL_STAT_OK),
		     ATTR_TYPE_END);
}

/* anvil_service_done - clean up */

static void anvil_service_done(VSTREAM *client_stream, char *unused_service,
			               char **unused_argv)
{
    ANVIL_LOCAL *anvil_local;
    const char *myname = "anvil_service_done";

    if (msg_verbose)
	msg_info("%s fd=%d stream=0x%lx",
		 myname, vstream_fileno(client_stream),
		 (unsigned long) client_stream);

    /*
     * Look up the local server, and get rid of any remote connection state
     * that we still have for this local server. Do not destroy remote client
     * status information before it expires.
     */
    if ((anvil_local = (ANVIL_LOCAL *) vstream_context(client_stream)) != 0) {
	if (msg_verbose)
	    msg_info("%s: anvil_local 0x%lx",
		     myname, (unsigned long) anvil_local);
	ANVIL_LOCAL_DROP_ALL(client_stream, anvil_local);
	myfree((void *) anvil_local);
    } else if (msg_verbose)
	msg_info("client socket not found for fd=%d",
		 vstream_fileno(client_stream));
}

/* anvil_status_dump - log and reset extreme usage */

static void anvil_status_dump(char *unused_name, char **unused_argv)
{
    ANVIL_MAX_RATE_REPORT(max_conn_rate, "connection");
    ANVIL_MAX_COUNT_REPORT(max_conn_count, "connection");
    ANVIL_MAX_RATE_REPORT(max_mail_rate, "message");
    ANVIL_MAX_RATE_REPORT(max_rcpt_rate, "recipient");
    ANVIL_MAX_RATE_REPORT(max_ntls_rate, "newtls");
    ANVIL_MAX_RATE_REPORT(max_auth_rate, "auth");

    if (max_cache_size > 0) {
	msg_info("statistics: max cache size %d at %.15s",
		 max_cache_size, ctime(&max_cache_time) + 4);
	max_cache_size = 0;
    }
}

/* anvil_status_update - log and reset extreme usage periodically */

static void anvil_status_update(int unused_event, void *context)
{
    anvil_status_dump((char *) 0, (char **) 0);
    event_request_timer(anvil_status_update, context, var_anvil_stat_time);
}

/* anvil_service - perform service for client */

static void anvil_service(VSTREAM *client_stream, char *unused_service, char **argv)
{
    static VSTRING *request;
    static VSTRING *ident;
    static const ANVIL_REQ_TABLE request_table[] = {
	ANVIL_REQ_CONN, anvil_remote_connect,
	ANVIL_REQ_MAIL, anvil_remote_mail,
	ANVIL_REQ_RCPT, anvil_remote_rcpt,
	ANVIL_REQ_NTLS, anvil_remote_newtls,
	ANVIL_REQ_DISC, anvil_remote_disconnect,
	ANVIL_REQ_NTLS_STAT, anvil_remote_newtls_stat,
	ANVIL_REQ_AUTH, anvil_remote_auth,
	ANVIL_REQ_LOOKUP, anvil_remote_lookup,
	0, 0,
    };
    const ANVIL_REQ_TABLE *rp;

    /*
     * Sanity check. This service takes no command-line arguments.
     */
    if (argv[0])
	msg_fatal("unexpected command-line argument: %s", argv[0]);

    /*
     * Initialize.
     */
    if (request == 0) {
	request = vstring_alloc(10);
	ident = vstring_alloc(10);
    }

    /*
     * This routine runs whenever a client connects to the socket dedicated
     * to the client connection rate management service. All
     * connection-management stuff is handled by the common code in
     * multi_server.c.
     */
    if (msg_verbose)
	msg_info("--- start request ---");
    if (attr_scan_plain(client_stream,
			ATTR_FLAG_MISSING | ATTR_FLAG_STRICT,
			RECV_ATTR_STR(ANVIL_ATTR_REQ, request),
			RECV_ATTR_STR(ANVIL_ATTR_IDENT, ident),
			ATTR_TYPE_END) == 2) {
	for (rp = request_table; /* see below */ ; rp++) {
	    if (rp->name == 0) {
		msg_warn("unrecognized request: \"%s\", ignored", STR(request));
		attr_print_plain(client_stream, ATTR_FLAG_NONE,
			  SEND_ATTR_INT(ANVIL_ATTR_STATUS, ANVIL_STAT_FAIL),
				 ATTR_TYPE_END);
		break;
	    }
	    if (STREQ(rp->name, STR(request))) {
		rp->action(client_stream, STR(ident));
		break;
	    }
	}
	vstream_fflush(client_stream);
    } else {
	/* Note: invokes anvil_service_done() */
	multi_server_disconnect(client_stream);
    }
    if (msg_verbose)
	msg_info("--- end request ---");
}

/* post_jail_init - post-jail initialization */

static void post_jail_init(char *unused_name, char **unused_argv)
{

    /*
     * Dump and reset extreme usage every so often.
     */
    event_request_timer(anvil_status_update, (void *) 0, var_anvil_stat_time);

    /*
     * Initial client state tables.
     */
    anvil_remote_map = htable_create(1000);

    /*
     * Do not limit the number of client requests.
     */
    var_use_limit = 0;

    /*
     * Don't exit before the sampling interval ends.
     */
    if (var_idle_limit < var_anvil_time_unit)
	var_idle_limit = var_anvil_time_unit;
}

MAIL_VERSION_STAMP_DECLARE;

/* main - pass control to the multi-threaded skeleton */

int     main(int argc, char **argv)
{
    static const CONFIG_TIME_TABLE time_table[] = {
	VAR_ANVIL_TIME_UNIT, DEF_ANVIL_TIME_UNIT, &var_anvil_time_unit, 1, 0,
	VAR_ANVIL_STAT_TIME, DEF_ANVIL_STAT_TIME, &var_anvil_stat_time, 1, 0,
	0,
    };

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    multi_server_main(argc, argv, anvil_service,
		      CA_MAIL_SERVER_TIME_TABLE(time_table),
		      CA_MAIL_SERVER_POST_INIT(post_jail_init),
		      CA_MAIL_SERVER_SOLITARY,
		      CA_MAIL_SERVER_PRE_DISCONN(anvil_service_done),
		      CA_MAIL_SERVER_EXIT(anvil_status_dump),
		      0);
}
