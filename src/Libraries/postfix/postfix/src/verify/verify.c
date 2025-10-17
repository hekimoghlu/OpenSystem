/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 2, 2024.
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
#include <sys/stat.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <htable.h>
#include <dict_ht.h>
#include <dict_cache.h>
#include <split_at.h>
#include <stringops.h>
#include <set_eugid.h>
#include <events.h>

/* Global library. */

#include <mail_conf.h>
#include <mail_params.h>
#include <mail_version.h>
#include <mail_proto.h>
#include <post_mail.h>
#include <data_redirect.h>
#include <verify_clnt.h>
#include <verify_sender_addr.h>

/* Server skeleton. */

#include <mail_server.h>

/* Application-specific. */

 /*
  * Tunable parameters.
  */
char   *var_verify_map;
int     var_verify_pos_exp;
int     var_verify_pos_try;
int     var_verify_neg_exp;
int     var_verify_neg_try;
int     var_verify_scan_cache;

 /*
  * State.
  */
static DICT_CACHE *verify_map;

 /*
  * Silly little macros.
  */
#define STR(x)			vstring_str(x)
#define STREQ(x,y)		(strcmp(x,y) == 0)

 /*
  * The address verification database consists of (address, data) tuples. The
  * format of the data field is "status:probed:updated:text". The meaning of
  * each field is:
  * 
  * status: one of the four recipient status codes (OK, DEFER, BOUNCE or TODO).
  * In the case of TODO, we have no information about the address, and the
  * address is being probed.
  * 
  * probed: if non-zero, the time the currently outstanding address probe was
  * sent. If zero, there is no outstanding address probe.
  * 
  * updated: if non-zero, the time the address probe result was received. If
  * zero, we have no information about the address, and the address is being
  * probed.
  * 
  * text: descriptive text from delivery agents etc.
  */

 /*
  * Quick test to see status without parsing the whole entry.
  */
#define STATUS_FROM_RAW_ENTRY(e) atoi(e)

/* verify_make_entry - construct table entry */

static void verify_make_entry(VSTRING *buf, int status, long probed,
			              long updated, const char *text)
{
    vstring_sprintf(buf, "%d:%ld:%ld:%s", status, probed, updated, text);
}

/* verify_parse_entry - parse table entry */

static int verify_parse_entry(char *buf, int *status, long *probed,
			              long *updated, char **text)
{
    char   *probed_text;
    char   *updated_text;

    if ((probed_text = split_at(buf, ':')) != 0
	&& (updated_text = split_at(probed_text, ':')) != 0
	&& (*text = split_at(updated_text, ':')) != 0
	&& alldig(buf)
	&& alldig(probed_text)
	&& alldig(updated_text)) {
	*probed = atol(probed_text);
	*updated = atol(updated_text);
	*status = atoi(buf);

	/*
	 * Coverity 200604: the code incorrectly tested (probed || updated),
	 * so that the sanity check never detected all-zero time stamps. Such
	 * records are never written. If we read a record with all-zero time
	 * stamps, then something is badly broken.
	 */
	if ((*status == DEL_RCPT_STAT_OK
	     || *status == DEL_RCPT_STAT_DEFER
	     || *status == DEL_RCPT_STAT_BOUNCE
	     || *status == DEL_RCPT_STAT_TODO)
	    && (*probed || *updated))
	    return (0);
    }
    msg_warn("bad address verify table entry: %.100s", buf);
    return (-1);
}

/* verify_stat2name - status to name */

static const char *verify_stat2name(int addr_status)
{
    if (addr_status == DEL_RCPT_STAT_OK)
	return ("deliverable");
    if (addr_status == DEL_RCPT_STAT_DEFER)
	return ("undeliverable");
    if (addr_status == DEL_RCPT_STAT_BOUNCE)
	return ("undeliverable");
    return (0);
}

/* verify_update_service - update address service */

static void verify_update_service(VSTREAM *client_stream)
{
    VSTRING *buf = vstring_alloc(10);
    VSTRING *addr = vstring_alloc(10);
    int     addr_status;
    VSTRING *text = vstring_alloc(10);
    const char *status_name;
    const char *raw_data;
    long    probed;
    long    updated;

    if (attr_scan(client_stream, ATTR_FLAG_STRICT,
		  RECV_ATTR_STR(MAIL_ATTR_ADDR, addr),
		  RECV_ATTR_INT(MAIL_ATTR_ADDR_STATUS, &addr_status),
		  RECV_ATTR_STR(MAIL_ATTR_WHY, text),
		  ATTR_TYPE_END) == 3) {
	/* FIX 200501 IPv6 patch did not neuter ":" in address literals. */
	translit(STR(addr), ":", "_");
	if ((status_name = verify_stat2name(addr_status)) == 0) {
	    msg_warn("bad recipient status %d for recipient %s",
		     addr_status, STR(addr));
	    attr_print(client_stream, ATTR_FLAG_NONE,
		       SEND_ATTR_INT(MAIL_ATTR_STATUS, VRFY_STAT_BAD),
		       ATTR_TYPE_END);
	} else {

	    /*
	     * Robustness: don't allow a failed probe to clobber an OK
	     * address before it expires. The failed probe is ignored so that
	     * the address will be re-probed upon the next query. As long as
	     * some probes succeed the address will remain cached as OK.
	     */
	    if (addr_status == DEL_RCPT_STAT_OK
		|| (raw_data = dict_cache_lookup(verify_map, STR(addr))) == 0
		|| STATUS_FROM_RAW_ENTRY(raw_data) != DEL_RCPT_STAT_OK) {
		probed = 0;
		updated = (long) time((time_t *) 0);
		verify_make_entry(buf, addr_status, probed, updated, STR(text));
		if (msg_verbose)
		    msg_info("PUT %s status=%d probed=%ld updated=%ld text=%s",
			STR(addr), addr_status, probed, updated, STR(text));
		dict_cache_update(verify_map, STR(addr), STR(buf));
	    }
	    attr_print(client_stream, ATTR_FLAG_NONE,
		       SEND_ATTR_INT(MAIL_ATTR_STATUS, VRFY_STAT_OK),
		       ATTR_TYPE_END);
	}
    }
    vstring_free(buf);
    vstring_free(addr);
    vstring_free(text);
}

/* verify_post_mail_fclose_action - callback */

static void verify_post_mail_fclose_action(int unused_status,
					           void *unused_context)
{
    /* no code here, we just need to avoid blocking in post_mail_fclose() */
}

/* verify_post_mail_action - callback */

static void verify_post_mail_action(VSTREAM *stream, void *context)
{

    /*
     * Probe messages need no body content, because they are never delivered,
     * deferred, or bounced.
     */
    if (stream != 0)
	post_mail_fclose_async(stream, verify_post_mail_fclose_action, context);
}

/* verify_query_service - query address status */

static void verify_query_service(VSTREAM *client_stream)
{
    VSTRING *addr = vstring_alloc(10);
    VSTRING *get_buf = 0;
    VSTRING *put_buf = 0;
    const char *raw_data;
    int     addr_status;
    long    probed;
    long    updated;
    char   *text;

    if (attr_scan(client_stream, ATTR_FLAG_STRICT,
		  RECV_ATTR_STR(MAIL_ATTR_ADDR, addr),
		  ATTR_TYPE_END) == 1) {
	long    now = (long) time((time_t *) 0);

	/*
	 * Produce a default record when no usable record exists.
	 * 
	 * If negative caching is disabled, purge an expired record from the
	 * database.
	 * 
	 * XXX Assume that a probe is lost if no response is received in 1000
	 * seconds. If this number is too small the queue will slowly fill up
	 * with delayed probes.
	 * 
	 * XXX Maintain a moving average for the probe turnaround time, and
	 * allow probe "retransmission" when a probe is outstanding for, say
	 * some minimal amount of time (1000 sec) plus several times the
	 * observed probe turnaround time. This causes probing to back off
	 * when the mail system becomes congested.
	 */
#define POSITIVE_ENTRY_EXPIRED(addr_status, updated) \
    (addr_status == DEL_RCPT_STAT_OK && updated + var_verify_pos_exp < now)
#define NEGATIVE_ENTRY_EXPIRED(addr_status, updated) \
    (addr_status != DEL_RCPT_STAT_OK && updated + var_verify_neg_exp < now)
#define PROBE_TTL	1000

	/* FIX 200501 IPv6 patch did not neuter ":" in address literals. */
	translit(STR(addr), ":", "_");
	if ((raw_data = dict_cache_lookup(verify_map, STR(addr))) == 0	/* not found */
	    || ((get_buf = vstring_alloc(10)),
		vstring_strcpy(get_buf, raw_data),	/* malformed */
		verify_parse_entry(STR(get_buf), &addr_status, &probed,
				   &updated, &text) < 0)
	    || (now - probed > PROBE_TTL	/* safe to probe */
		&& (POSITIVE_ENTRY_EXPIRED(addr_status, updated)
		    || NEGATIVE_ENTRY_EXPIRED(addr_status, updated)))) {
	    addr_status = DEL_RCPT_STAT_TODO;
	    probed = 0;
	    updated = 0;
	    text = "Address verification in progress";
	    if (raw_data != 0 && var_verify_neg_cache == 0)
		dict_cache_delete(verify_map, STR(addr));
	}
	if (msg_verbose)
	    msg_info("GOT %s status=%d probed=%ld updated=%ld text=%s",
		     STR(addr), addr_status, probed, updated, text);

	/*
	 * Respond to the client.
	 */
	attr_print(client_stream, ATTR_FLAG_NONE,
		   SEND_ATTR_INT(MAIL_ATTR_STATUS, VRFY_STAT_OK),
		   SEND_ATTR_INT(MAIL_ATTR_ADDR_STATUS, addr_status),
		   SEND_ATTR_STR(MAIL_ATTR_WHY, text),
		   ATTR_TYPE_END);

	/*
	 * Send a new probe when the information needs to be refreshed.
	 * 
	 * XXX For an initial proof of concept implementation, use synchronous
	 * mail submission. This needs to be made async for high-volume
	 * sites, which makes it even more interesting to eliminate duplicate
	 * queries while a probe is being built.
	 * 
	 * If negative caching is turned off, update the database only when
	 * refreshing an existing entry.
	 */
#define POSITIVE_REFRESH_NEEDED(addr_status, updated) \
    (addr_status == DEL_RCPT_STAT_OK && updated + var_verify_pos_try < now)
#define NEGATIVE_REFRESH_NEEDED(addr_status, updated) \
    (addr_status != DEL_RCPT_STAT_OK && updated + var_verify_neg_try < now)

	if (now - probed > PROBE_TTL
	       && (POSITIVE_REFRESH_NEEDED(addr_status, updated)
		   || NEGATIVE_REFRESH_NEEDED(addr_status, updated))) {
	    if (msg_verbose)
		msg_info("PROBE %s status=%d probed=%ld updated=%ld",
			 STR(addr), addr_status, now, updated);
	    post_mail_fopen_async(make_verify_sender_addr(), STR(addr),
				  MAIL_SRC_MASK_VERIFY,
				  DEL_REQ_FLAG_MTA_VRFY,
				  SMTPUTF8_FLAG_NONE,
				  (VSTRING *) 0,
				  verify_post_mail_action,
				  (void *) 0);
	    if (updated != 0 || var_verify_neg_cache != 0) {
		put_buf = vstring_alloc(10);
		verify_make_entry(put_buf, addr_status, now, updated, text);
		if (msg_verbose)
		    msg_info("PUT %s status=%d probed=%ld updated=%ld text=%s",
			     STR(addr), addr_status, now, updated, text);
		dict_cache_update(verify_map, STR(addr), STR(put_buf));
	    }
	}
    }
    vstring_free(addr);
    if (get_buf)
	vstring_free(get_buf);
    if (put_buf)
	vstring_free(put_buf);
}

/* verify_cache_validator - cache cleanup validator */

static int verify_cache_validator(const char *addr, const char *raw_data,
				          void *context)
{
    VSTRING *get_buf = (VSTRING *) context;
    int     addr_status;
    long    probed;
    long    updated;
    char   *text;
    long    now = (long) event_time();

#define POS_OR_NEG_ENTRY_EXPIRED(stat, stamp) \
	(POSITIVE_ENTRY_EXPIRED((stat), (stamp)) \
	    || NEGATIVE_ENTRY_EXPIRED((stat), (stamp)))

    vstring_strcpy(get_buf, raw_data);
    return (verify_parse_entry(STR(get_buf), &addr_status,	/* syntax OK */
			       &probed, &updated, &text) == 0
	    && (now - probed < PROBE_TTL	/* probe in progress */
		|| !POS_OR_NEG_ENTRY_EXPIRED(addr_status, updated)));
}

/* verify_service - perform service for client */

static void verify_service(VSTREAM *client_stream, char *unused_service,
			           char **argv)
{
    VSTRING *request = vstring_alloc(10);

    /*
     * Sanity check. This service takes no command-line arguments.
     */
    if (argv[0])
	msg_fatal("unexpected command-line argument: %s", argv[0]);

    /*
     * This routine runs whenever a client connects to the socket dedicated
     * to the address verification service. All connection-management stuff
     * is handled by the common code in multi_server.c.
     */
    if (attr_scan(client_stream,
		  ATTR_FLAG_MORE | ATTR_FLAG_STRICT,
		  RECV_ATTR_STR(MAIL_ATTR_REQ, request),
		  ATTR_TYPE_END) == 1) {
	if (STREQ(STR(request), VRFY_REQ_UPDATE)) {
	    verify_update_service(client_stream);
	} else if (STREQ(STR(request), VRFY_REQ_QUERY)) {
	    verify_query_service(client_stream);
	} else {
	    msg_warn("unrecognized request: \"%s\", ignored", STR(request));
	    attr_print(client_stream, ATTR_FLAG_NONE,
		       SEND_ATTR_INT(MAIL_ATTR_STATUS, VRFY_STAT_BAD),
		       ATTR_TYPE_END);
	}
    }
    vstream_fflush(client_stream);
    vstring_free(request);
}

/* verify_dump - dump some statistics */

static void verify_dump(char *unused_name, char **unused_argv)
{

    /*
     * Dump preliminary cache cleanup statistics when the process commits
     * suicide while a cache cleanup run is in progress. We can't currently
     * distinguish between "postfix reload" (we should restart) or "maximal
     * idle time reached" (we could finish the cache cleanup first).
     */
    dict_cache_close(verify_map);
    verify_map = 0;
}

/* post_jail_init - post-jail initialization */

static void post_jail_init(char *unused_name, char **unused_argv)
{

    /*
     * If the database is in volatile memory only, prevent automatic process
     * suicide after a limited number of client requests or after a limited
     * amount of idle time.
     */
    if (*var_verify_map == 0) {
	var_use_limit = 0;
	var_idle_limit = 0;
    }

    /*
     * Start the cache cleanup thread.
     */
    if (var_verify_scan_cache > 0) {
	int     cache_flags;

	cache_flags = DICT_CACHE_FLAG_STATISTICS;
	if (msg_verbose)
	    cache_flags |= DICT_CACHE_FLAG_VERBOSE;
	dict_cache_control(verify_map,
			   CA_DICT_CACHE_CTL_FLAGS(cache_flags),
			   CA_DICT_CACHE_CTL_INTERVAL(var_verify_scan_cache),
			CA_DICT_CACHE_CTL_VALIDATOR(verify_cache_validator),
		     CA_DICT_CACHE_CTL_CONTEXT((void *) vstring_alloc(100)),
			   CA_DICT_CACHE_CTL_END);
    }
}

/* pre_jail_init - pre-jail initialization */

static void pre_jail_init(char *unused_name, char **unused_argv)
{
    mode_t  saved_mask;
    VSTRING *redirect;

    /*
     * Never, ever, get killed by a master signal, as that would corrupt the
     * database when we're in the middle of an update.
     */
    setsid();

    /*
     * Security: don't create root-owned files that contain untrusted data.
     * And don't create Postfix-owned files in root-owned directories,
     * either. We want a correct relationship between (file/directory)
     * ownership and (file/directory) content.
     * 
     * XXX Non-root open can violate the principle of least surprise: Postfix
     * can't open an *SQL config file for database read-write access, even
     * though it can open that same control file for database read-only
     * access.
     * 
     * The solution is to query a map type and obtain its properties before
     * opening it. A clean solution is to add a dict_info() API that is
     * similar to dict_open() except it returns properties (dict flags) only.
     * A pragmatic solution is to overload the existing API and have
     * dict_open() return a dummy map when given a null map name.
     * 
     * However, the proxymap daemon has been opening *SQL maps as non-root for
     * years now without anyone complaining, let's not solve a problem that
     * doesn't exist.
     */
    SAVE_AND_SET_EUGID(var_owner_uid, var_owner_gid);
    redirect = vstring_alloc(100);

    /*
     * Keep state in persistent (external) or volatile (internal) map.
     * 
     * Start the cache cleanup thread after permanently dropping privileges.
     */
#define VERIFY_DICT_OPEN_FLAGS (DICT_FLAG_DUP_REPLACE | DICT_FLAG_SYNC_UPDATE \
	    | DICT_FLAG_OPEN_LOCK | DICT_FLAG_UTF8_REQUEST)

    saved_mask = umask(022);
    verify_map =
	dict_cache_open(*var_verify_map ?
			data_redirect_map(redirect, var_verify_map) :
			"internal:verify",
			O_CREAT | O_RDWR, VERIFY_DICT_OPEN_FLAGS);
    (void) umask(saved_mask);

    /*
     * Clean up and restore privilege.
     */
    vstring_free(redirect);
    RESTORE_SAVED_EUGID();
}

MAIL_VERSION_STAMP_DECLARE;

/* main - pass control to the multi-threaded skeleton */

int     main(int argc, char **argv)
{
    static const CONFIG_STR_TABLE str_table[] = {
	VAR_VERIFY_MAP, DEF_VERIFY_MAP, &var_verify_map, 0, 0,
	VAR_VERIFY_SENDER, DEF_VERIFY_SENDER, &var_verify_sender, 0, 0,
	0,
    };
    static const CONFIG_TIME_TABLE time_table[] = {
	VAR_VERIFY_POS_EXP, DEF_VERIFY_POS_EXP, &var_verify_pos_exp, 1, 0,
	VAR_VERIFY_POS_TRY, DEF_VERIFY_POS_TRY, &var_verify_pos_try, 1, 0,
	VAR_VERIFY_NEG_EXP, DEF_VERIFY_NEG_EXP, &var_verify_neg_exp, 1, 0,
	VAR_VERIFY_NEG_TRY, DEF_VERIFY_NEG_TRY, &var_verify_neg_try, 1, 0,
	VAR_VERIFY_SCAN_CACHE, DEF_VERIFY_SCAN_CACHE, &var_verify_scan_cache, 0, 0,
	VAR_VERIFY_SENDER_TTL, DEF_VERIFY_SENDER_TTL, &var_verify_sender_ttl, 0, 0,
	0,
    };

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    multi_server_main(argc, argv, verify_service,
		      CA_MAIL_SERVER_STR_TABLE(str_table),
		      CA_MAIL_SERVER_TIME_TABLE(time_table),
		      CA_MAIL_SERVER_PRE_INIT(pre_jail_init),
		      CA_MAIL_SERVER_POST_INIT(post_jail_init),
		      CA_MAIL_SERVER_SOLITARY,
		      CA_MAIL_SERVER_EXIT(verify_dump),
		      0);
}
