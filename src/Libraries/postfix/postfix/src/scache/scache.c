/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 26, 2022.
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
#include <time.h>

/* Utility library. */

#include <msg.h>
#include <iostuff.h>
#include <htable.h>
#include <ring.h>
#include <events.h>

/* Global library. */

#include <mail_params.h>
#include <mail_version.h>
#include <mail_proto.h>
#include <scache.h>

/* Single server skeleton. */

#include <mail_server.h>
#include <mail_conf.h>

/* Application-specific. */

 /*
  * Tunable parameters.
  */
int     var_scache_ttl_lim;
int     var_scache_stat_time;

 /*
  * Request parameters.
  */
static VSTRING *scache_request;
static VSTRING *scache_dest_label;
static VSTRING *scache_dest_prop;
static VSTRING *scache_endp_label;
static VSTRING *scache_endp_prop;

#ifdef CANT_WRITE_BEFORE_SENDING_FD
static VSTRING *scache_dummy;

#endif

 /*
  * Session cache instance.
  */
static SCACHE *scache;

 /*
  * Statistics.
  */
static int scache_dest_hits;
static int scache_dest_miss;
static int scache_dest_count;
static int scache_endp_hits;
static int scache_endp_miss;
static int scache_endp_count;
static int scache_sess_count;
time_t  scache_start_time;

 /*
  * Silly little macros.
  */
#define STR(x)			vstring_str(x)
#define VSTREQ(x,y)		(strcmp(STR(x),y) == 0)

/* scache_save_endp_service - protocol to save endpoint->stream binding */

static void scache_save_endp_service(VSTREAM *client_stream)
{
    const char *myname = "scache_save_endp_service";
    int     ttl;
    int     fd;
    SCACHE_SIZE size;

    if (attr_scan(client_stream,
		  ATTR_FLAG_STRICT,
		  RECV_ATTR_INT(MAIL_ATTR_TTL, &ttl),
		  RECV_ATTR_STR(MAIL_ATTR_LABEL, scache_endp_label),
		  RECV_ATTR_STR(MAIL_ATTR_PROP, scache_endp_prop),
		  ATTR_TYPE_END) != 3
	|| ttl <= 0) {
	msg_warn("%s: bad or missing request parameter", myname);
	attr_print(client_stream, ATTR_FLAG_NONE,
		   SEND_ATTR_INT(MAIL_ATTR_STATUS, SCACHE_STAT_BAD),
		   ATTR_TYPE_END);
	return;
    } else if (
#ifdef CANT_WRITE_BEFORE_SENDING_FD
	       attr_print(client_stream, ATTR_FLAG_NONE,
			  SEND_ATTR_STR(MAIL_ATTR_DUMMY, ""),
			  ATTR_TYPE_END) != 0
	       || vstream_fflush(client_stream) != 0
	       || read_wait(vstream_fileno(client_stream),
			    client_stream->timeout) < 0	/* XXX */
	       ||
#endif
	       (fd = LOCAL_RECV_FD(vstream_fileno(client_stream))) < 0) {
	msg_warn("%s: unable to receive file descriptor: %m", myname);
	(void) attr_print(client_stream, ATTR_FLAG_NONE,
			  SEND_ATTR_INT(MAIL_ATTR_STATUS, SCACHE_STAT_FAIL),
			  ATTR_TYPE_END);
	return;
    } else {
	scache_save_endp(scache,
			 ttl > var_scache_ttl_lim ? var_scache_ttl_lim : ttl,
			 STR(scache_endp_label), STR(scache_endp_prop), fd);
	(void) attr_print(client_stream, ATTR_FLAG_NONE,
			  SEND_ATTR_INT(MAIL_ATTR_STATUS, SCACHE_STAT_OK),
			  ATTR_TYPE_END);
	scache_size(scache, &size);
	if (size.endp_count > scache_endp_count)
	    scache_endp_count = size.endp_count;
	if (size.sess_count > scache_sess_count)
	    scache_sess_count = size.sess_count;
	return;
    }
}

/* scache_find_endp_service - protocol to find connection for endpoint */

static void scache_find_endp_service(VSTREAM *client_stream)
{
    const char *myname = "scache_find_endp_service";
    int     fd;

    if (attr_scan(client_stream,
		  ATTR_FLAG_STRICT,
		  RECV_ATTR_STR(MAIL_ATTR_LABEL, scache_endp_label),
		  ATTR_TYPE_END) != 1) {
	msg_warn("%s: bad or missing request parameter", myname);
	attr_print(client_stream, ATTR_FLAG_NONE,
		   SEND_ATTR_INT(MAIL_ATTR_STATUS, SCACHE_STAT_BAD),
		   SEND_ATTR_STR(MAIL_ATTR_PROP, ""),
		   ATTR_TYPE_END);
	return;
    } else if ((fd = scache_find_endp(scache, STR(scache_endp_label),
				      scache_endp_prop)) < 0) {
	attr_print(client_stream, ATTR_FLAG_NONE,
		   SEND_ATTR_INT(MAIL_ATTR_STATUS, SCACHE_STAT_FAIL),
		   SEND_ATTR_STR(MAIL_ATTR_PROP, ""),
		   ATTR_TYPE_END);
	scache_endp_miss++;
	return;
    } else {
	attr_print(client_stream, ATTR_FLAG_NONE,
		   SEND_ATTR_INT(MAIL_ATTR_STATUS, SCACHE_STAT_OK),
		   SEND_ATTR_STR(MAIL_ATTR_PROP, STR(scache_endp_prop)),
		   ATTR_TYPE_END);
	if (vstream_fflush(client_stream) != 0
#ifdef CANT_WRITE_BEFORE_SENDING_FD
	    || attr_scan(client_stream, ATTR_FLAG_STRICT,
			 RECV_ATTR_STR(MAIL_ATTR_DUMMY, scache_dummy),
			 ATTR_TYPE_END) != 1
#endif
	    || LOCAL_SEND_FD(vstream_fileno(client_stream), fd) < 0
#ifdef MUST_READ_AFTER_SENDING_FD
	    || attr_scan(client_stream, ATTR_FLAG_STRICT,
			 RECV_ATTR_STR(MAIL_ATTR_DUMMY, scache_dummy),
			 ATTR_TYPE_END) != 1
#endif
	    )
	    msg_warn("%s: cannot send file descriptor: %m", myname);
	if (close(fd) < 0)
	    msg_warn("close(%d): %m", fd);
	scache_endp_hits++;
	return;
    }
}

/* scache_save_dest_service - protocol to save destination->endpoint binding */

static void scache_save_dest_service(VSTREAM *client_stream)
{
    const char *myname = "scache_save_dest_service";
    int     ttl;
    SCACHE_SIZE size;

    if (attr_scan(client_stream,
		  ATTR_FLAG_STRICT,
		  RECV_ATTR_INT(MAIL_ATTR_TTL, &ttl),
		  RECV_ATTR_STR(MAIL_ATTR_LABEL, scache_dest_label),
		  RECV_ATTR_STR(MAIL_ATTR_PROP, scache_dest_prop),
		  RECV_ATTR_STR(MAIL_ATTR_LABEL, scache_endp_label),
		  ATTR_TYPE_END) != 4
	|| ttl <= 0) {
	msg_warn("%s: bad or missing request parameter", myname);
	attr_print(client_stream, ATTR_FLAG_NONE,
		   SEND_ATTR_INT(MAIL_ATTR_STATUS, SCACHE_STAT_BAD),
		   ATTR_TYPE_END);
	return;
    } else {
	scache_save_dest(scache,
			 ttl > var_scache_ttl_lim ? var_scache_ttl_lim : ttl,
			 STR(scache_dest_label), STR(scache_dest_prop),
			 STR(scache_endp_label));
	attr_print(client_stream, ATTR_FLAG_NONE,
		   SEND_ATTR_INT(MAIL_ATTR_STATUS, SCACHE_STAT_OK),
		   ATTR_TYPE_END);
	scache_size(scache, &size);
	if (size.dest_count > scache_dest_count)
	    scache_dest_count = size.dest_count;
	if (size.endp_count > scache_endp_count)
	    scache_endp_count = size.endp_count;
	return;
    }
}

/* scache_find_dest_service - protocol to find connection for destination */

static void scache_find_dest_service(VSTREAM *client_stream)
{
    const char *myname = "scache_find_dest_service";
    int     fd;

    if (attr_scan(client_stream,
		  ATTR_FLAG_STRICT,
		  RECV_ATTR_STR(MAIL_ATTR_LABEL, scache_dest_label),
		  ATTR_TYPE_END) != 1) {
	msg_warn("%s: bad or missing request parameter", myname);
	attr_print(client_stream, ATTR_FLAG_NONE,
		   SEND_ATTR_INT(MAIL_ATTR_STATUS, SCACHE_STAT_BAD),
		   SEND_ATTR_STR(MAIL_ATTR_PROP, ""),
		   SEND_ATTR_STR(MAIL_ATTR_PROP, ""),
		   ATTR_TYPE_END);
	return;
    } else if ((fd = scache_find_dest(scache, STR(scache_dest_label),
				      scache_dest_prop,
				      scache_endp_prop)) < 0) {
	attr_print(client_stream, ATTR_FLAG_NONE,
		   SEND_ATTR_INT(MAIL_ATTR_STATUS, SCACHE_STAT_FAIL),
		   SEND_ATTR_STR(MAIL_ATTR_PROP, ""),
		   SEND_ATTR_STR(MAIL_ATTR_PROP, ""),
		   ATTR_TYPE_END);
	scache_dest_miss++;
	return;
    } else {
	attr_print(client_stream, ATTR_FLAG_NONE,
		   SEND_ATTR_INT(MAIL_ATTR_STATUS, SCACHE_STAT_OK),
		   SEND_ATTR_STR(MAIL_ATTR_PROP, STR(scache_dest_prop)),
		   SEND_ATTR_STR(MAIL_ATTR_PROP, STR(scache_endp_prop)),
		   ATTR_TYPE_END);
	if (vstream_fflush(client_stream) != 0
#ifdef CANT_WRITE_BEFORE_SENDING_FD
	    || attr_scan(client_stream, ATTR_FLAG_STRICT,
			 RECV_ATTR_STR(MAIL_ATTR_DUMMY, scache_dummy),
			 ATTR_TYPE_END) != 1
#endif
	    || LOCAL_SEND_FD(vstream_fileno(client_stream), fd) < 0
#ifdef MUST_READ_AFTER_SENDING_FD
	    || attr_scan(client_stream, ATTR_FLAG_STRICT,
			 RECV_ATTR_STR(MAIL_ATTR_DUMMY, scache_dummy),
			 ATTR_TYPE_END) != 1
#endif
	    )
	    msg_warn("%s: cannot send file descriptor: %m", myname);
	if (close(fd) < 0)
	    msg_warn("close(%d): %m", fd);
	scache_dest_hits++;
	return;
    }
}

/* scache_service - perform service for client */

static void scache_service(VSTREAM *client_stream, char *unused_service,
			           char **argv)
{

    /*
     * Sanity check. This service takes no command-line arguments.
     */
    if (argv[0])
	msg_fatal("unexpected command-line argument: %s", argv[0]);

    /*
     * This routine runs whenever a client connects to the UNIX-domain socket
     * dedicated to the scache service. All connection-management stuff is
     * handled by the common code in multi_server.c.
     * 
     * XXX Workaround: with some requests, the client sends a dummy message
     * after the server replies (yes that's a botch). When the scache server
     * is slow, this dummy message may become concatenated with the next
     * request from the same client. The do-while loop below will repeat
     * instead of discarding the client request. We must process it now
     * because there will be no select() notification.
     */
    do {
	if (attr_scan(client_stream,
		      ATTR_FLAG_MORE | ATTR_FLAG_STRICT,
		      RECV_ATTR_STR(MAIL_ATTR_REQ, scache_request),
		      ATTR_TYPE_END) == 1) {
	    if (VSTREQ(scache_request, SCACHE_REQ_SAVE_DEST)) {
		scache_save_dest_service(client_stream);
	    } else if (VSTREQ(scache_request, SCACHE_REQ_FIND_DEST)) {
		scache_find_dest_service(client_stream);
	    } else if (VSTREQ(scache_request, SCACHE_REQ_SAVE_ENDP)) {
		scache_save_endp_service(client_stream);
	    } else if (VSTREQ(scache_request, SCACHE_REQ_FIND_ENDP)) {
		scache_find_endp_service(client_stream);
	    } else {
		msg_warn("unrecognized request: \"%s\", ignored",
			 STR(scache_request));
		attr_print(client_stream, ATTR_FLAG_NONE,
			   SEND_ATTR_INT(MAIL_ATTR_STATUS, SCACHE_STAT_BAD),
			   ATTR_TYPE_END);
	    }
	}
    } while (vstream_peek(client_stream) > 0);
    vstream_fflush(client_stream);
}

/* scache_status_dump - log and reset cache statistics */

static void scache_status_dump(char *unused_name, char **unused_argv)
{
    if (scache_dest_hits || scache_dest_miss
	|| scache_endp_hits || scache_endp_miss
	|| scache_dest_count || scache_endp_count
	|| scache_sess_count)
	msg_info("statistics: start interval %.15s",
		 ctime(&scache_start_time) + 4);

    if (scache_dest_hits || scache_dest_miss) {
	msg_info("statistics: domain lookup hits=%d miss=%d success=%d%%",
		 scache_dest_hits, scache_dest_miss,
		 scache_dest_hits * 100
		 / (scache_dest_hits + scache_dest_miss));
	scache_dest_hits = scache_dest_miss = 0;
    }
    if (scache_endp_hits || scache_endp_miss) {
	msg_info("statistics: address lookup hits=%d miss=%d success=%d%%",
		 scache_endp_hits, scache_endp_miss,
		 scache_endp_hits * 100
		 / (scache_endp_hits + scache_endp_miss));
	scache_endp_hits = scache_endp_miss = 0;
    }
    if (scache_dest_count || scache_endp_count || scache_sess_count) {
	msg_info("statistics: max simultaneous domains=%d addresses=%d connection=%d",
		 scache_dest_count, scache_endp_count, scache_sess_count);
	scache_dest_count = 0;
	scache_endp_count = 0;
	scache_sess_count = 0;
    }
    scache_start_time = event_time();
}

/* scache_status_update - log and reset cache statistics periodically */

static void scache_status_update(int unused_event, void *context)
{
    scache_status_dump((char *) 0, (char **) 0);
    event_request_timer(scache_status_update, context, var_scache_stat_time);
}

/* post_jail_init - initialization after privilege drop */

static void post_jail_init(char *unused_name, char **unused_argv)
{

    /*
     * Pre-allocate the cache instance.
     */
    scache = scache_multi_create();

    /*
     * Pre-allocate buffers.
     */
    scache_request = vstring_alloc(10);
    scache_dest_label = vstring_alloc(10);
    scache_dest_prop = vstring_alloc(10);
    scache_endp_label = vstring_alloc(10);
    scache_endp_prop = vstring_alloc(10);
#ifdef CANT_WRITE_BEFORE_SENDING_FD
    scache_dummy = vstring_alloc(10);
#endif

    /*
     * Disable the max_use limit. We still terminate when no client is
     * connected for $idle_limit time units.
     */
    var_use_limit = 0;

    /*
     * Dump and reset cache statistics every so often.
     */
    event_request_timer(scache_status_update, (void *) 0, var_scache_stat_time);
    scache_start_time = event_time();
}

MAIL_VERSION_STAMP_DECLARE;

/* main - pass control to the multi-threaded skeleton */

int     main(int argc, char **argv)
{
    static const CONFIG_TIME_TABLE time_table[] = {
	VAR_SCACHE_TTL_LIM, DEF_SCACHE_TTL_LIM, &var_scache_ttl_lim, 1, 0,
	VAR_SCACHE_STAT_TIME, DEF_SCACHE_STAT_TIME, &var_scache_stat_time, 1, 0,
	0,
    };

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    multi_server_main(argc, argv, scache_service,
		      CA_MAIL_SERVER_TIME_TABLE(time_table),
		      CA_MAIL_SERVER_POST_INIT(post_jail_init),
		      CA_MAIL_SERVER_EXIT(scache_status_dump),
		      CA_MAIL_SERVER_SOLITARY,
		      0);
}
