/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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
#include <fcntl.h>
#ifdef USE_PATHS_H
#include <paths.h>
#endif

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <htable.h>
#include <vstring.h>
#include <vstream.h>
#include <iostuff.h>
#include <name_mask.h>
#include <set_eugid.h>
#include <dict.h>

/* Global library. */

#include <recipient_list.h>
#include <deliver_request.h>
#include <deliver_completed.h>
#include <mail_params.h>
#include <mail_addr.h>
#include <mail_conf.h>
#include <been_here.h>
#include <mail_params.h>
#include <mail_version.h>
#include <ext_prop.h>
#include <maps.h>
#include <flush_clnt.h>

/* Single server skeleton. */

#include <mail_server.h>

/* Application-specific. */

#include "local.h"

 /*
  * Tunable parameters.
  */
char   *var_allow_commands;
char   *var_allow_files;
char   *var_alias_maps;
int     var_dup_filter_limit;
int     var_command_maxtime;		/* You can now leave this here. */
char   *var_home_mailbox;
char   *var_mailbox_command;
char   *var_mailbox_cmd_maps;
char   *var_rcpt_fdelim;
char   *var_local_cmd_shell;
char   *var_luser_relay;
int     var_biff;
char   *var_mail_spool_dir;
char   *var_mailbox_transport;
char   *var_mbox_transp_maps;
char   *var_fallback_transport;
char   *var_fbck_transp_maps;
char   *var_exec_directory;
char   *var_exec_exp_filter;
char   *var_forward_path;
char   *var_cmd_exp_filter;
char   *var_fwd_exp_filter;
char   *var_prop_extension;
int     var_exp_own_alias;
char   *var_deliver_hdr;
int     var_stat_home_dir;
int     var_mailtool_compat;
char   *var_mailbox_lock;
long    var_mailbox_limit;
bool    var_frozen_delivered;
bool    var_reset_owner_attr;
bool    var_strict_mbox_owner;

int     local_cmd_deliver_mask;
int     local_file_deliver_mask;
int     local_ext_prop_mask;
int     local_deliver_hdr_mask;
int     local_mbox_lock_mask;
MAPS   *alias_maps;
char   *var_local_dsn_filter;

/* local_deliver - deliver message with extreme prejudice */

static int local_deliver(DELIVER_REQUEST *rqst, char *service)
{
    const char *myname = "local_deliver";
    RECIPIENT *rcpt_end = rqst->rcpt_list.info + rqst->rcpt_list.len;
    RECIPIENT *rcpt;
    int     rcpt_stat;
    int     msg_stat;
    LOCAL_STATE state;
    USER_ATTR usr_attr;

    if (msg_verbose)
	msg_info("local_deliver: %s from %s", rqst->queue_id, rqst->sender);

    /*
     * Initialize the delivery attributes that are not recipient specific.
     * While messages are being delivered and while aliases or forward files
     * are being expanded, this attribute list is being changed constantly.
     * For this reason, the list is passed on by value (except when it is
     * being initialized :-), so that there is no need to undo attribute
     * changes made by lower-level routines. The alias/include/forward
     * expansion attribute list is part of a tree with self and parent
     * references (see the EXPAND_ATTR definitions). The user-specific
     * attributes are security sensitive, and are therefore kept separate.
     * All this results in a noticeable level of clumsiness, but passing
     * things around by value gives good protection against accidental change
     * by subroutines.
     */
    state.level = 0;
    deliver_attr_init(&state.msg_attr);
    state.msg_attr.queue_name = rqst->queue_name;
    state.msg_attr.queue_id = rqst->queue_id;
    state.msg_attr.fp = rqst->fp;
    state.msg_attr.offset = rqst->data_offset;
    state.msg_attr.encoding = rqst->encoding;
    state.msg_attr.smtputf8 = rqst->smtputf8;
    state.msg_attr.sender = rqst->sender;
    state.msg_attr.dsn_envid = rqst->dsn_envid;
    state.msg_attr.dsn_ret = rqst->dsn_ret;
    state.msg_attr.relay = service;
    state.msg_attr.msg_stats = rqst->msg_stats;
    state.msg_attr.request = rqst;
    RESET_OWNER_ATTR(state.msg_attr, state.level);
    RESET_USER_ATTR(usr_attr, state.level);
    state.loop_info = delivered_hdr_init(rqst->fp, rqst->data_offset,
					 FOLD_ADDR_ALL);
    state.request = rqst;

    /*
     * Iterate over each recipient named in the delivery request. When the
     * mail delivery status for a given recipient is definite (i.e. bounced
     * or delivered), update the message queue file and cross off the
     * recipient. Update the per-message delivery status.
     */
    for (msg_stat = 0, rcpt = rqst->rcpt_list.info; rcpt < rcpt_end; rcpt++) {
	state.dup_filter = been_here_init(var_dup_filter_limit, BH_FLAG_FOLD);
	forward_init();
	state.msg_attr.rcpt = *rcpt;
	rcpt_stat = deliver_recipient(state, usr_attr);
	rcpt_stat |= forward_finish(rqst, state.msg_attr, rcpt_stat);
	if (rcpt_stat == 0 && (rqst->flags & DEL_REQ_FLAG_SUCCESS))
	    deliver_completed(state.msg_attr.fp, rcpt->offset);
	been_here_free(state.dup_filter);
	msg_stat |= rcpt_stat;
    }

    /*
     * Clean up.
     */
    delivered_hdr_free(state.loop_info);
    deliver_attr_free(&state.msg_attr);

    return (msg_stat);
}

/* local_service - perform service for client */

static void local_service(VSTREAM *stream, char *service, char **argv)
{
    DELIVER_REQUEST *request;
    int     status;

    /*
     * Sanity check. This service takes no command-line arguments.
     */
    if (argv[0])
	msg_fatal("unexpected command-line argument: %s", argv[0]);

    /*
     * This routine runs whenever a client connects to the UNIX-domain socket
     * that is dedicated to local mail delivery service. What we see below is
     * a little protocol to (1) tell the client that we are ready, (2) read a
     * delivery request from the client, and (3) report the completion status
     * of that request.
     */
    if ((request = deliver_request_read(stream)) != 0) {
	status = local_deliver(request, service);
	deliver_request_done(stream, request, status);
    }
}

/* local_mask_init - initialize delivery restrictions */

static void local_mask_init(void)
{
    static const NAME_MASK file_mask[] = {
	"alias", EXPAND_TYPE_ALIAS,
	"forward", EXPAND_TYPE_FWD,
	"include", EXPAND_TYPE_INCL,
	0,
    };
    static const NAME_MASK command_mask[] = {
	"alias", EXPAND_TYPE_ALIAS,
	"forward", EXPAND_TYPE_FWD,
	"include", EXPAND_TYPE_INCL,
	0,
    };
    static const NAME_MASK deliver_mask[] = {
	"command", DELIVER_HDR_CMD,
	"file", DELIVER_HDR_FILE,
	"forward", DELIVER_HDR_FWD,
	0,
    };

    local_file_deliver_mask = name_mask(VAR_ALLOW_FILES, file_mask,
					var_allow_files);
    local_cmd_deliver_mask = name_mask(VAR_ALLOW_COMMANDS, command_mask,
				       var_allow_commands);
    local_ext_prop_mask =
	ext_prop_mask(VAR_PROP_EXTENSION, var_prop_extension);
    local_deliver_hdr_mask = name_mask(VAR_DELIVER_HDR, deliver_mask,
				       var_deliver_hdr);
    local_mbox_lock_mask = mbox_lock_mask(var_mailbox_lock);
    if (var_mailtool_compat) {
	msg_warn("%s: deprecated parameter, use \"%s = dotlock\" instead",
		 VAR_MAILTOOL_COMPAT, VAR_MAILBOX_LOCK);
	local_mbox_lock_mask &= MBOX_DOT_LOCK;
    }
    if (local_mbox_lock_mask == 0)
	msg_fatal("parameter %s specifies no applicable mailbox locking method",
		  VAR_MAILBOX_LOCK);
}

/* pre_accept - see if tables have changed */

static void pre_accept(char *unused_name, char **unused_argv)
{
    const char *table;

    if ((table = dict_changed_name()) != 0) {
	msg_info("table %s has changed -- restarting", table);
	exit(0);
    }
}

/* post_init - post-jail initialization */

static void post_init(char *unused_name, char **unused_argv)
{

    /*
     * Drop privileges most of the time, and set up delivery restrictions.
     */
    set_eugid(var_owner_uid, var_owner_gid);
    local_mask_init();
}

/* pre_init - pre-jail initialization */

static void pre_init(char *unused_name, char **unused_argv)
{

    /*
     * Reset the file size limit from the message size limit to the mailbox
     * size limit. XXX This still isn't accurate because the file size limit
     * also affects delivery to command.
     * 
     * A file size limit protects the machine against runaway software errors.
     * It is not suitable to enforce mail quota, because users can get around
     * mail quota by delivering to /file/name or to |command.
     * 
     * We can't have mailbox size limit smaller than the message size limit,
     * because that prohibits the delivery agent from updating the queue
     * file.
     */
    if (var_mailbox_limit) {
	if (var_mailbox_limit < var_message_limit || var_message_limit == 0)
	    msg_fatal("main.cf configuration error: %s is smaller than %s",
		      VAR_MAILBOX_LIMIT, VAR_MESSAGE_LIMIT);
	set_file_limit(var_mailbox_limit);
    }
    alias_maps = maps_create("aliases", var_alias_maps,
			     DICT_FLAG_LOCK | DICT_FLAG_PARANOID
			     | DICT_FLAG_FOLD_FIX
			     | DICT_FLAG_UTF8_REQUEST);

    flush_init();
}

MAIL_VERSION_STAMP_DECLARE;

/* main - pass control to the single-threaded skeleton */

int     main(int argc, char **argv)
{
    static const CONFIG_TIME_TABLE time_table[] = {
	VAR_COMMAND_MAXTIME, DEF_COMMAND_MAXTIME, &var_command_maxtime, 1, 0,
	0,
    };
    static const CONFIG_INT_TABLE int_table[] = {
	VAR_DUP_FILTER_LIMIT, DEF_DUP_FILTER_LIMIT, &var_dup_filter_limit, 0, 0,
	0,
    };
    static const CONFIG_LONG_TABLE long_table[] = {
	VAR_MAILBOX_LIMIT, DEF_MAILBOX_LIMIT, &var_mailbox_limit, 0, 0,
	0,
    };
    static const CONFIG_STR_TABLE str_table[] = {
	VAR_ALIAS_MAPS, DEF_ALIAS_MAPS, &var_alias_maps, 0, 0,
	VAR_HOME_MAILBOX, DEF_HOME_MAILBOX, &var_home_mailbox, 0, 0,
	VAR_ALLOW_COMMANDS, DEF_ALLOW_COMMANDS, &var_allow_commands, 0, 0,
	VAR_ALLOW_FILES, DEF_ALLOW_FILES, &var_allow_files, 0, 0,
	VAR_LOCAL_CMD_SHELL, DEF_LOCAL_CMD_SHELL, &var_local_cmd_shell, 0, 0,
	VAR_MAIL_SPOOL_DIR, DEF_MAIL_SPOOL_DIR, &var_mail_spool_dir, 0, 0,
	VAR_MAILBOX_TRANSP, DEF_MAILBOX_TRANSP, &var_mailbox_transport, 0, 0,
	VAR_MBOX_TRANSP_MAPS, DEF_MBOX_TRANSP_MAPS, &var_mbox_transp_maps, 0, 0,
	VAR_FALLBACK_TRANSP, DEF_FALLBACK_TRANSP, &var_fallback_transport, 0, 0,
	VAR_FBCK_TRANSP_MAPS, DEF_FBCK_TRANSP_MAPS, &var_fbck_transp_maps, 0, 0,
	VAR_CMD_EXP_FILTER, DEF_CMD_EXP_FILTER, &var_cmd_exp_filter, 1, 0,
	VAR_FWD_EXP_FILTER, DEF_FWD_EXP_FILTER, &var_fwd_exp_filter, 1, 0,
	VAR_EXEC_EXP_FILTER, DEF_EXEC_EXP_FILTER, &var_exec_exp_filter, 1, 0,
	VAR_PROP_EXTENSION, DEF_PROP_EXTENSION, &var_prop_extension, 0, 0,
	VAR_DELIVER_HDR, DEF_DELIVER_HDR, &var_deliver_hdr, 0, 0,
	VAR_MAILBOX_LOCK, DEF_MAILBOX_LOCK, &var_mailbox_lock, 1, 0,
	VAR_MAILBOX_CMD_MAPS, DEF_MAILBOX_CMD_MAPS, &var_mailbox_cmd_maps, 0, 0,
	VAR_LOCAL_DSN_FILTER, DEF_LOCAL_DSN_FILTER, &var_local_dsn_filter, 0, 0,
	0,
    };
    static const CONFIG_BOOL_TABLE bool_table[] = {
	VAR_BIFF, DEF_BIFF, &var_biff,
	VAR_EXP_OWN_ALIAS, DEF_EXP_OWN_ALIAS, &var_exp_own_alias,
	VAR_STAT_HOME_DIR, DEF_STAT_HOME_DIR, &var_stat_home_dir,
	VAR_MAILTOOL_COMPAT, DEF_MAILTOOL_COMPAT, &var_mailtool_compat,
	VAR_FROZEN_DELIVERED, DEF_FROZEN_DELIVERED, &var_frozen_delivered,
	VAR_RESET_OWNER_ATTR, DEF_RESET_OWNER_ATTR, &var_reset_owner_attr,
	VAR_STRICT_MBOX_OWNER, DEF_STRICT_MBOX_OWNER, &var_strict_mbox_owner,
	0,
    };

    /* Suppress $name expansion upon loading. */
    static const CONFIG_RAW_TABLE raw_table[] = {
	VAR_EXEC_DIRECTORY, DEF_EXEC_DIRECTORY, &var_exec_directory, 0, 0,
	VAR_FORWARD_PATH, DEF_FORWARD_PATH, &var_forward_path, 0, 0,
	VAR_MAILBOX_COMMAND, DEF_MAILBOX_COMMAND, &var_mailbox_command, 0, 0,
	VAR_LUSER_RELAY, DEF_LUSER_RELAY, &var_luser_relay, 0, 0,
	0,
    };

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    single_server_main(argc, argv, local_service,
		       CA_MAIL_SERVER_INT_TABLE(int_table),
		       CA_MAIL_SERVER_LONG_TABLE(long_table),
		       CA_MAIL_SERVER_STR_TABLE(str_table),
		       CA_MAIL_SERVER_RAW_TABLE(raw_table),
		       CA_MAIL_SERVER_BOOL_TABLE(bool_table),
		       CA_MAIL_SERVER_TIME_TABLE(time_table),
		       CA_MAIL_SERVER_PRE_INIT(pre_init),
		       CA_MAIL_SERVER_POST_INIT(post_init),
		       CA_MAIL_SERVER_PRE_ACCEPT(pre_accept),
		       CA_MAIL_SERVER_PRIVILEGED,
		       CA_MAIL_SERVER_BOUNCE_INIT(VAR_LOCAL_DSN_FILTER,
						  &var_local_dsn_filter),
		       0);
}
