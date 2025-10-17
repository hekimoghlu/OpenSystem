/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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
#include <stdlib.h>
#include <string.h>

 /*
  * Utility library
  */
#include <msg.h>
#include <mymalloc.h>
#include <stringops.h>
#include <split_at.h>

 /*
  * Global library
  */
#include <mail_params.h>
#include <string_list.h>
#include <maps.h>
#include <mail_addr_find.h>
#include <smtp_stream.h>

 /*
  * XSASL library.
  */
#include <xsasl.h>

 /*
  * Application-specific
  */
#include "smtp.h"
#include "smtp_sasl.h"
#include "smtp_sasl_auth_cache.h"

#ifdef USE_SASL_AUTH

 /*
  * Per-host login/password information.
  */
static MAPS *smtp_sasl_passwd_map;

 /*
  * Supported SASL mechanisms.
  */
STRING_LIST *smtp_sasl_mechs;

 /*
  * SASL implementation handle.
  */
static XSASL_CLIENT_IMPL *smtp_sasl_impl;

 /*
  * The 535 SASL authentication failure cache.
  */
#ifdef HAVE_SASL_AUTH_CACHE
static SMTP_SASL_AUTH_CACHE *smtp_sasl_auth_cache;

#endif

/* smtp_sasl_passwd_lookup - password lookup routine */

int     smtp_sasl_passwd_lookup(SMTP_SESSION *session)
{
    const char *myname = "smtp_sasl_passwd_lookup";
    SMTP_STATE *state = session->state;
    SMTP_ITERATOR *iter = session->iterator;
    const char *value;
    char   *passwd;

    /*
     * Sanity check.
     */
    if (smtp_sasl_passwd_map == 0)
	msg_panic("%s: passwd map not initialized", myname);

    /*
     * Look up the per-server password information. Try the hostname first,
     * then try the destination.
     * 
     * XXX Instead of using nexthop (the intended destination) we use dest
     * (either the intended destination, or a fall-back destination).
     * 
     * XXX SASL authentication currently depends on the host/domain but not on
     * the TCP port. If the port is not :25, we should append it to the table
     * lookup key. Code for this was briefly introduced into 2.2 snapshots,
     * but didn't canonicalize the TCP port, and did not append the port to
     * the MX hostname.
     */
    smtp_sasl_passwd_map->error = 0;
    if ((smtp_mode
	 && var_smtp_sender_auth && state->request->sender[0]
	 && (value = mail_addr_find(smtp_sasl_passwd_map,
				 state->request->sender, (char **) 0)) != 0)
	|| (smtp_sasl_passwd_map->error == 0
	    && (value = maps_find(smtp_sasl_passwd_map,
				  STR(iter->host), 0)) != 0)
	|| (smtp_sasl_passwd_map->error == 0
	    && (value = maps_find(smtp_sasl_passwd_map,
				  STR(iter->dest), 0)) != 0)) {
	if (session->sasl_username)
	    myfree(session->sasl_username);
	session->sasl_username = mystrdup(value);
	passwd = split_at(session->sasl_username, ':');
	if (session->sasl_passwd)
	    myfree(session->sasl_passwd);
	session->sasl_passwd = mystrdup(passwd ? passwd : "");
	if (msg_verbose)
	    msg_info("%s: host `%s' user `%s' pass `%s'",
		     myname, STR(iter->host),
		     session->sasl_username, session->sasl_passwd);
	return (1);
    } else if (smtp_sasl_passwd_map->error) {
	msg_warn("%s: %s lookup error",
		 state->request->queue_id, smtp_sasl_passwd_map->title);
	vstream_longjmp(session->stream, SMTP_ERR_DATA);
    } else {
	if (msg_verbose)
	    msg_info("%s: no auth info found (sender=`%s', host=`%s')",
		     myname, state->request->sender, STR(iter->host));
	return (0);
    }
}

/* smtp_sasl_initialize - per-process initialization (pre jail) */

void    smtp_sasl_initialize(void)
{

    /*
     * Sanity check.
     */
    if (smtp_sasl_passwd_map || smtp_sasl_impl)
	msg_panic("smtp_sasl_initialize: repeated call");
    if (*var_smtp_sasl_passwd == 0)
	msg_fatal("specify a password table via the `%s' configuration parameter",
		  VAR_LMTP_SMTP(SASL_PASSWD));

    /*
     * Open the per-host password table and initialize the SASL library. Use
     * shared locks for reading, just in case someone updates the table.
     */
    smtp_sasl_passwd_map = maps_create(VAR_LMTP_SMTP(SASL_PASSWD),
				       var_smtp_sasl_passwd,
				       DICT_FLAG_LOCK | DICT_FLAG_FOLD_FIX
				       | DICT_FLAG_UTF8_REQUEST);
    if ((smtp_sasl_impl = xsasl_client_init(var_smtp_sasl_type,
					    var_smtp_sasl_path)) == 0)
	msg_fatal("SASL library initialization");

    /*
     * Initialize optional supported mechanism matchlist
     */
    if (*var_smtp_sasl_mechs)
	smtp_sasl_mechs = string_list_init(VAR_SMTP_SASL_MECHS,
					   MATCH_FLAG_NONE,
					   var_smtp_sasl_mechs);

    /*
     * Initialize the 535 SASL authentication failure cache.
     */
    if (*var_smtp_sasl_auth_cache_name) {
#ifdef HAVE_SASL_AUTH_CACHE
	smtp_sasl_auth_cache =
	    smtp_sasl_auth_cache_init(var_smtp_sasl_auth_cache_name,
				      var_smtp_sasl_auth_cache_time);
#else
	msg_warn("not compiled with TLS support -- "
	    "ignoring the %s setting", VAR_LMTP_SMTP(SASL_AUTH_CACHE_NAME));
#endif
    }
}

/* smtp_sasl_connect - per-session client initialization */

void    smtp_sasl_connect(SMTP_SESSION *session)
{

    /*
     * This initialization happens whenever we instantiate an SMTP session
     * object. We don't instantiate a SASL client until we actually need one.
     */
    session->sasl_mechanism_list = 0;
    session->sasl_username = 0;
    session->sasl_passwd = 0;
    session->sasl_client = 0;
    session->sasl_reply = 0;
}

/* smtp_sasl_start - per-session SASL initialization */

void    smtp_sasl_start(SMTP_SESSION *session, const char *sasl_opts_name,
			        const char *sasl_opts_val)
{
    XSASL_CLIENT_CREATE_ARGS create_args;
    SMTP_ITERATOR *iter = session->iterator;

    if (msg_verbose)
	msg_info("starting new SASL client");
    if ((session->sasl_client =
	 XSASL_CLIENT_CREATE(smtp_sasl_impl, &create_args,
			     stream = session->stream,
			     service = var_procname,
			     server_name = STR(iter->host),
			     security_options = sasl_opts_val)) == 0)
	msg_fatal("SASL per-connection initialization failed");
    session->sasl_reply = vstring_alloc(20);
}

/* smtp_sasl_authenticate - run authentication protocol */

int     smtp_sasl_authenticate(SMTP_SESSION *session, DSN_BUF *why)
{
    const char *myname = "smtp_sasl_authenticate";
    SMTP_ITERATOR *iter = session->iterator;
    SMTP_RESP *resp;
    const char *mechanism;
    int     result;
    char   *line;
    int     steps = 0;

    /*
     * Sanity check.
     */
    if (session->sasl_mechanism_list == 0)
	msg_panic("%s: no mechanism list", myname);

    if (msg_verbose)
	msg_info("%s: %s: SASL mechanisms %s",
		 myname, session->namaddrport, session->sasl_mechanism_list);

    /*
     * Avoid repeated login failures after a recent 535 error.
     */
#ifdef HAVE_SASL_AUTH_CACHE
    if (smtp_sasl_auth_cache
	&& smtp_sasl_auth_cache_find(smtp_sasl_auth_cache, session)) {
	char   *resp_dsn = smtp_sasl_auth_cache_dsn(smtp_sasl_auth_cache);
	char   *resp_str = smtp_sasl_auth_cache_text(smtp_sasl_auth_cache);

	if (var_smtp_sasl_auth_soft_bounce && resp_dsn[0] == '5')
	    resp_dsn[0] = '4';
	dsb_update(why, resp_dsn, DSB_DEF_ACTION, DSB_MTYPE_DNS,
		   STR(iter->host), var_procname, resp_str,
		   "SASL [CACHED] authentication failed; server %s said: %s",
		   STR(iter->host), resp_str);
	return (0);
    }
#endif

    /*
     * Start the client side authentication protocol.
     */
    result = xsasl_client_first(session->sasl_client,
				session->sasl_mechanism_list,
				session->sasl_username,
				session->sasl_passwd,
				&mechanism, session->sasl_reply);
    if (result != XSASL_AUTH_OK) {
	dsb_update(why, "4.7.0", DSB_DEF_ACTION, DSB_SKIP_RMTA,
		   DSB_DTYPE_SASL, STR(session->sasl_reply),
		   "SASL authentication failed; "
		   "cannot authenticate to server %s: %s",
		   session->namaddr, STR(session->sasl_reply));
	return (-1);
    }

    /*
     * Send the AUTH command and the optional initial client response.
     * sasl_encode64() produces four bytes for each complete or incomplete
     * triple of input bytes. Allocate an extra byte for string termination.
     */
    if (LEN(session->sasl_reply) > 0) {
	smtp_chat_cmd(session, "AUTH %s %s", mechanism,
		      STR(session->sasl_reply));
    } else {
	smtp_chat_cmd(session, "AUTH %s", mechanism);
    }

    /*
     * Step through the authentication protocol until the server tells us
     * that we are done.
     */
    while ((resp = smtp_chat_resp(session))->code / 100 == 3) {

	/*
	 * Sanity check.
	 */
	if (++steps > 100) {
	    dsb_simple(why, "4.3.0", "SASL authentication failed; "
		       "authentication protocol loop with server %s",
		       session->namaddr);
	    return (-1);
	}

	/*
	 * Process a server challenge.
	 */
	line = resp->str;
	(void) mystrtok(&line, "- \t\n");	/* skip over result code */
	result = xsasl_client_next(session->sasl_client, line,
				   session->sasl_reply);
	if (result != XSASL_AUTH_OK) {
	    dsb_update(why, "4.7.0", DSB_DEF_ACTION,	/* Fix 200512 */
		    DSB_SKIP_RMTA, DSB_DTYPE_SASL, STR(session->sasl_reply),
		       "SASL authentication failed; "
		       "cannot authenticate to server %s: %s",
		       session->namaddr, STR(session->sasl_reply));
	    return (-1);			/* Fix 200512 */
	}

	/*
	 * Send a client response.
	 */
	smtp_chat_cmd(session, "%s", STR(session->sasl_reply));
    }

    /*
     * We completed the authentication protocol.
     */
    if (resp->code / 100 != 2) {
#ifdef HAVE_SASL_AUTH_CACHE
	/* Update the 535 authentication failure cache. */
	if (smtp_sasl_auth_cache && resp->code == 535)
	    smtp_sasl_auth_cache_store(smtp_sasl_auth_cache, session, resp);
#endif
	if (var_smtp_sasl_auth_soft_bounce && resp->code / 100 == 5)
	    STR(resp->dsn_buf)[0] = '4';
	dsb_update(why, resp->dsn, DSB_DEF_ACTION,
		   DSB_MTYPE_DNS, STR(iter->host),
		   var_procname, resp->str,
		   "SASL authentication failed; server %s said: %s",
		   session->namaddr, resp->str);
	return (0);
    }
    return (1);
}

/* smtp_sasl_cleanup - per-session cleanup */

void    smtp_sasl_cleanup(SMTP_SESSION *session)
{
    if (session->sasl_username) {
	myfree(session->sasl_username);
	session->sasl_username = 0;
    }
    if (session->sasl_passwd) {
	myfree(session->sasl_passwd);
	session->sasl_passwd = 0;
    }
    if (session->sasl_mechanism_list) {
	/* allocated in smtp_sasl_helo_auth */
	myfree(session->sasl_mechanism_list);
	session->sasl_mechanism_list = 0;
    }
    if (session->sasl_client) {
	if (msg_verbose)
	    msg_info("disposing SASL state information");
	xsasl_client_free(session->sasl_client);
	session->sasl_client = 0;
    }
    if (session->sasl_reply) {
	vstring_free(session->sasl_reply);
	session->sasl_reply = 0;
    }
}

/* smtp_sasl_passivate - append serialized SASL attributes */

void    smtp_sasl_passivate(SMTP_SESSION *session, VSTRING *buf)
{
}

/* smtp_sasl_activate - de-serialize SASL attributes */

int     smtp_sasl_activate(SMTP_SESSION *session, char *buf)
{
    return (0);
}

#endif
