/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 27, 2025.
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
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <vstring.h>
#include <vstream.h>
#include <vstring_vstream.h>
#include <split_at.h>
#include <stringops.h>
#include <dict.h>
#include <events.h>

/* Global library. */

#include <mail_params.h>
#include <mail_version.h>
#include <mail_proto.h>
#include <resolve_local.h>
#include <mail_conf.h>
#include <resolve_clnt.h>
#include <rewrite_clnt.h>
#include <tok822.h>
#include <mail_addr.h>

/* Multi server skeleton. */

#include <mail_server.h>

/* Application-specific. */

#include <trivial-rewrite.h>
#include <transport.h>

static VSTRING *command;

 /*
  * Tunable parameters.
  */
char   *var_transport_maps;
bool    var_swap_bangpath;
bool    var_append_dot_mydomain;
bool    var_append_at_myorigin;
bool    var_percent_hack;
char   *var_local_transport;
char   *var_virt_transport;
char   *var_relay_transport;
int     var_resolve_dequoted;
char   *var_virt_alias_maps;		/* XXX virtual_alias_domains */
char   *var_virt_mailbox_maps;		/* XXX virtual_mailbox_domains */
char   *var_virt_alias_doms;
char   *var_virt_mailbox_doms;
char   *var_relocated_maps;
char   *var_def_transport;
char   *var_snd_def_xport_maps;
char   *var_empty_addr;
int     var_show_unk_rcpt_table;
int     var_resolve_nulldom;
char   *var_remote_rwr_domain;
char   *var_snd_relay_maps;
char   *var_null_relay_maps_key;
char   *var_null_def_xport_maps_key;
int     var_resolve_num_dom;
bool    var_allow_min_user;

 /*
  * Shadow personality for address verification.
  */
char   *var_vrfy_xport_maps;
char   *var_vrfy_local_xport;
char   *var_vrfy_virt_xport;
char   *var_vrfy_relay_xport;
char   *var_vrfy_def_xport;
char   *var_vrfy_snd_def_xport_maps;
char   *var_vrfy_relayhost;
char   *var_vrfy_relay_maps;

 /*
  * Different resolver personalities depending on the kind of request.
  */
RES_CONTEXT resolve_regular = {
    VAR_LOCAL_TRANSPORT, &var_local_transport,
    VAR_VIRT_TRANSPORT, &var_virt_transport,
    VAR_RELAY_TRANSPORT, &var_relay_transport,
    VAR_DEF_TRANSPORT, &var_def_transport,
    VAR_SND_DEF_XPORT_MAPS, &var_snd_def_xport_maps, 0,
    VAR_RELAYHOST, &var_relayhost,
    VAR_SND_RELAY_MAPS, &var_snd_relay_maps, 0,
    VAR_TRANSPORT_MAPS, &var_transport_maps, 0
};

RES_CONTEXT resolve_verify = {
    VAR_VRFY_LOCAL_XPORT, &var_vrfy_local_xport,
    VAR_VRFY_VIRT_XPORT, &var_vrfy_virt_xport,
    VAR_VRFY_RELAY_XPORT, &var_vrfy_relay_xport,
    VAR_VRFY_DEF_XPORT, &var_vrfy_def_xport,
    VAR_VRFY_SND_DEF_XPORT_MAPS, &var_vrfy_snd_def_xport_maps, 0,
    VAR_VRFY_RELAYHOST, &var_vrfy_relayhost,
    VAR_VRFY_RELAY_MAPS, &var_vrfy_relay_maps, 0,
    VAR_VRFY_XPORT_MAPS, &var_vrfy_xport_maps, 0
};

 /*
  * Connection management. When file-based lookup tables change we should
  * restart at our convenience, but avoid client read errors. We restart
  * rather than reopen, because the process may be chrooted (and if it isn't
  * we still need code that handles the chrooted case anyway).
  * 
  * Three variants are implemented. Only one should be used.
  * 
  * ifdef DETACH_AND_ASK_CLIENTS_TO_RECONNECT
  * 
  * This code detaches the trivial-rewrite process from the master, stops
  * accepting new clients, and handles established clients in the background,
  * asking them to reconnect the next time they send a request. The master
  * creates a new process that accepts connections. This is reasonably safe
  * because the number of trivial-rewrite server processes is small compared
  * to the number of trivial-rewrite client processes. The few extra
  * background processes should not make a difference in Postfix's footprint.
  * However, once a daemon detaches from the master, its exit status will be
  * lost, and abnormal termination may remain undetected. Timely restart is
  * achieved by checking the table changed status every 10 seconds or so
  * before responding to a client request.
  * 
  * ifdef CHECK_TABLE_STATS_PERIODICALLY
  * 
  * This code runs every 10 seconds and terminates the process when lookup
  * tables have changed. This is subject to race conditions when established
  * clients send a request while the server exits; those clients may read EOF
  * instead of a server reply. If the experience with the oldest option
  * (below) is anything to go by, however, then this is unlikely to be a
  * problem during real deployment.
  * 
  * ifdef CHECK_TABLE_STATS_BEFORE_ACCEPT
  * 
  * This is the old code. It checks the table changed status when a new client
  * connects (i.e. before the server calls accept()), and terminates
  * immediately. This is invisible for the connecting client, but is subject
  * to race conditions when established clients send a request while the
  * server exits; those clients may read EOF instead of a server reply. This
  * has, however, not been a problem in real deployment. With the old code,
  * timely restart is achieved by setting the ipc_ttl parameter to 60
  * seconds, so that the table change status is checked several times a
  * minute.
  */
int     server_flags;

 /*
  * Define exactly one of these.
  */
/* #define DETACH_AND_ASK_CLIENTS_TO_RECONNECT	/* correct and complex */
#define CHECK_TABLE_STATS_PERIODICALLY	/* quick */
/* #define CHECK_TABLE_STATS_BEFORE_ACCEPT	/* slow */

/* rewrite_service - read request and send reply */

static void rewrite_service(VSTREAM *stream, char *unused_service, char **argv)
{
    int     status = -1;

#ifdef DETACH_AND_ASK_CLIENTS_TO_RECONNECT
    static time_t last;
    time_t  now;
    const char *table;

#endif

    /*
     * Sanity check. This service takes no command-line arguments.
     */
    if (argv[0])
	msg_fatal("unexpected command-line argument: %s", argv[0]);

    /*
     * Client connections are long-lived. Be sure to refesh timely.
     */
#ifdef DETACH_AND_ASK_CLIENTS_TO_RECONNECT
    if (server_flags == 0 && (now = event_time()) - last > 10) {
	if ((table = dict_changed_name()) != 0) {
	    msg_info("table %s has changed -- restarting", table);
	    if (multi_server_drain() == 0)
		server_flags = 1;
	}
	last = now;
    }
#endif

    /*
     * This routine runs whenever a client connects to the UNIX-domain socket
     * dedicated to address rewriting. All connection-management stuff is
     * handled by the common code in multi_server.c.
     */
    if (attr_scan(stream, ATTR_FLAG_STRICT | ATTR_FLAG_MORE,
		  RECV_ATTR_STR(MAIL_ATTR_REQ, command),
		  ATTR_TYPE_END) == 1) {
	if (strcmp(vstring_str(command), REWRITE_ADDR) == 0) {
	    status = rewrite_proto(stream);
	} else if (strcmp(vstring_str(command), RESOLVE_REGULAR) == 0) {
	    status = resolve_proto(&resolve_regular, stream);
	} else if (strcmp(vstring_str(command), RESOLVE_VERIFY) == 0) {
	    status = resolve_proto(&resolve_verify, stream);
	} else {
	    msg_warn("bad command %.30s", printable(vstring_str(command), '?'));
	}
    }
    if (status < 0)
	multi_server_disconnect(stream);
}

/* pre_accept - see if tables have changed */

#ifdef CHECK_TABLE_STATS_BEFORE_ACCEPT

static void pre_accept(char *unused_name, char **unused_argv)
{
    const char *table;

    if ((table = dict_changed_name()) != 0) {
	msg_info("table %s has changed -- restarting", table);
	exit(0);
    }
}

#endif

static void check_table_stats(int unused_event, void *unused_context)
{
    const char *table;

    if ((table = dict_changed_name()) != 0) {
	msg_info("table %s has changed -- restarting", table);
	exit(0);
    }
    event_request_timer(check_table_stats, (void *) 0, 10);
}

/* pre_jail_init - initialize before entering chroot jail */

static void pre_jail_init(char *unused_name, char **unused_argv)
{
    command = vstring_alloc(100);
    rewrite_init();
    resolve_init();
    if (*RES_PARAM_VALUE(resolve_regular.transport_maps))
	resolve_regular.transport_info =
	    transport_pre_init(resolve_regular.transport_maps_name,
			   RES_PARAM_VALUE(resolve_regular.transport_maps));
    if (*RES_PARAM_VALUE(resolve_verify.transport_maps))
	resolve_verify.transport_info =
	    transport_pre_init(resolve_verify.transport_maps_name,
			    RES_PARAM_VALUE(resolve_verify.transport_maps));
    if (*RES_PARAM_VALUE(resolve_regular.snd_relay_maps))
	resolve_regular.snd_relay_info =
	    maps_create(resolve_regular.snd_relay_maps_name,
			RES_PARAM_VALUE(resolve_regular.snd_relay_maps),
			DICT_FLAG_LOCK | DICT_FLAG_FOLD_FIX
			| DICT_FLAG_NO_REGSUB | DICT_FLAG_UTF8_REQUEST);
    if (*RES_PARAM_VALUE(resolve_verify.snd_relay_maps))
	resolve_verify.snd_relay_info =
	    maps_create(resolve_verify.snd_relay_maps_name,
			RES_PARAM_VALUE(resolve_verify.snd_relay_maps),
			DICT_FLAG_LOCK | DICT_FLAG_FOLD_FIX
			| DICT_FLAG_NO_REGSUB | DICT_FLAG_UTF8_REQUEST);
    if (*RES_PARAM_VALUE(resolve_regular.snd_def_xp_maps))
	resolve_regular.snd_def_xp_info =
	    maps_create(resolve_regular.snd_def_xp_maps_name,
			RES_PARAM_VALUE(resolve_regular.snd_def_xp_maps),
			DICT_FLAG_LOCK | DICT_FLAG_FOLD_FIX
			| DICT_FLAG_NO_REGSUB | DICT_FLAG_UTF8_REQUEST);
    if (*RES_PARAM_VALUE(resolve_verify.snd_def_xp_maps))
	resolve_verify.snd_def_xp_info =
	    maps_create(resolve_verify.snd_def_xp_maps_name,
			RES_PARAM_VALUE(resolve_verify.snd_def_xp_maps),
			DICT_FLAG_LOCK | DICT_FLAG_FOLD_FIX
			| DICT_FLAG_NO_REGSUB | DICT_FLAG_UTF8_REQUEST);
}

/* post_jail_init - initialize after entering chroot jail */

static void post_jail_init(char *unused_name, char **unused_argv)
{
    if (resolve_regular.transport_info)
	transport_post_init(resolve_regular.transport_info);
    if (resolve_verify.transport_info)
	transport_post_init(resolve_verify.transport_info);
    check_table_stats(0, (void *) 0);
}

MAIL_VERSION_STAMP_DECLARE;

/* main - pass control to the multi-threaded skeleton code */

int     main(int argc, char **argv)
{
    static const CONFIG_STR_TABLE str_table[] = {
	VAR_TRANSPORT_MAPS, DEF_TRANSPORT_MAPS, &var_transport_maps, 0, 0,
	VAR_LOCAL_TRANSPORT, DEF_LOCAL_TRANSPORT, &var_local_transport, 1, 0,
	VAR_VIRT_TRANSPORT, DEF_VIRT_TRANSPORT, &var_virt_transport, 1, 0,
	VAR_RELAY_TRANSPORT, DEF_RELAY_TRANSPORT, &var_relay_transport, 1, 0,
	VAR_DEF_TRANSPORT, DEF_DEF_TRANSPORT, &var_def_transport, 1, 0,
	VAR_VIRT_ALIAS_MAPS, DEF_VIRT_ALIAS_MAPS, &var_virt_alias_maps, 0, 0,
	VAR_VIRT_ALIAS_DOMS, DEF_VIRT_ALIAS_DOMS, &var_virt_alias_doms, 0, 0,
	VAR_VIRT_MAILBOX_MAPS, DEF_VIRT_MAILBOX_MAPS, &var_virt_mailbox_maps, 0, 0,
	VAR_VIRT_MAILBOX_DOMS, DEF_VIRT_MAILBOX_DOMS, &var_virt_mailbox_doms, 0, 0,
	VAR_RELOCATED_MAPS, DEF_RELOCATED_MAPS, &var_relocated_maps, 0, 0,
	VAR_EMPTY_ADDR, DEF_EMPTY_ADDR, &var_empty_addr, 1, 0,
	VAR_VRFY_XPORT_MAPS, DEF_VRFY_XPORT_MAPS, &var_vrfy_xport_maps, 0, 0,
	VAR_VRFY_LOCAL_XPORT, DEF_VRFY_LOCAL_XPORT, &var_vrfy_local_xport, 1, 0,
	VAR_VRFY_VIRT_XPORT, DEF_VRFY_VIRT_XPORT, &var_vrfy_virt_xport, 1, 0,
	VAR_VRFY_RELAY_XPORT, DEF_VRFY_RELAY_XPORT, &var_vrfy_relay_xport, 1, 0,
	VAR_VRFY_DEF_XPORT, DEF_VRFY_DEF_XPORT, &var_vrfy_def_xport, 1, 0,
	VAR_VRFY_RELAYHOST, DEF_VRFY_RELAYHOST, &var_vrfy_relayhost, 0, 0,
	VAR_REM_RWR_DOMAIN, DEF_REM_RWR_DOMAIN, &var_remote_rwr_domain, 0, 0,
	VAR_SND_RELAY_MAPS, DEF_SND_RELAY_MAPS, &var_snd_relay_maps, 0, 0,
	VAR_NULL_RELAY_MAPS_KEY, DEF_NULL_RELAY_MAPS_KEY, &var_null_relay_maps_key, 1, 0,
	VAR_VRFY_RELAY_MAPS, DEF_VRFY_RELAY_MAPS, &var_vrfy_relay_maps, 0, 0,
	VAR_SND_DEF_XPORT_MAPS, DEF_SND_DEF_XPORT_MAPS, &var_snd_def_xport_maps, 0, 0,
	VAR_NULL_DEF_XPORT_MAPS_KEY, DEF_NULL_DEF_XPORT_MAPS_KEY, &var_null_def_xport_maps_key, 1, 0,
	VAR_VRFY_SND_DEF_XPORT_MAPS, DEF_VRFY_SND_DEF_XPORT_MAPS, &var_vrfy_snd_def_xport_maps, 0, 0,
	0,
    };
    static const CONFIG_BOOL_TABLE bool_table[] = {
	VAR_SWAP_BANGPATH, DEF_SWAP_BANGPATH, &var_swap_bangpath,
	VAR_APP_AT_MYORIGIN, DEF_APP_AT_MYORIGIN, &var_append_at_myorigin,
	VAR_PERCENT_HACK, DEF_PERCENT_HACK, &var_percent_hack,
	VAR_RESOLVE_DEQUOTED, DEF_RESOLVE_DEQUOTED, &var_resolve_dequoted,
	VAR_SHOW_UNK_RCPT_TABLE, DEF_SHOW_UNK_RCPT_TABLE, &var_show_unk_rcpt_table,
	VAR_RESOLVE_NULLDOM, DEF_RESOLVE_NULLDOM, &var_resolve_nulldom,
	VAR_RESOLVE_NUM_DOM, DEF_RESOLVE_NUM_DOM, &var_resolve_num_dom,
	VAR_ALLOW_MIN_USER, DEF_ALLOW_MIN_USER, &var_allow_min_user,
	0,
    };
    static const CONFIG_NBOOL_TABLE nbool_table[] = {
	VAR_APP_DOT_MYDOMAIN, DEF_APP_DOT_MYDOMAIN, &var_append_dot_mydomain,
	0,
    };

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    multi_server_main(argc, argv, rewrite_service,
		      CA_MAIL_SERVER_STR_TABLE(str_table),
		      CA_MAIL_SERVER_BOOL_TABLE(bool_table),
		      CA_MAIL_SERVER_NBOOL_TABLE(nbool_table),
		      CA_MAIL_SERVER_PRE_INIT(pre_jail_init),
		      CA_MAIL_SERVER_POST_INIT(post_jail_init),
#ifdef CHECK_TABLE_STATS_BEFORE_ACCEPT
		      CA_MAIL_SERVER_PRE_ACCEPT(pre_accept),
#endif
		      0);
}
