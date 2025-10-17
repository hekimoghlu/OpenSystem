/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 26, 2023.
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
#include <stdlib.h>
#ifdef USE_PATHS_H
#include <paths.h>			/* XXX mail_spool_dir dependency */
#endif

/* Utility library. */

#include <msg.h>
#include <vstring.h>
#include <vstream.h>
#include <iostuff.h>
#include <set_eugid.h>
#include <dict.h>

/* Global library. */

#include <mail_queue.h>
#include <recipient_list.h>
#include <deliver_request.h>
#include <deliver_completed.h>
#include <mail_params.h>
#include <mail_version.h>
#include <mail_conf.h>
#include <mail_params.h>
#include <mail_addr_find.h>
#include <flush_clnt.h>

/* Single server skeleton. */

#include <mail_server.h>

/* Application-specific. */

#include "virtual.h"

 /*
  * Tunable parameters.
  */
char   *var_virt_mailbox_maps;
char   *var_virt_uid_maps;
char   *var_virt_gid_maps;
int     var_virt_minimum_uid;
char   *var_virt_mailbox_base;
char   *var_virt_mailbox_lock;
long    var_virt_mailbox_limit;
char   *var_mail_spool_dir;		/* XXX dependency fix */
bool    var_strict_mbox_owner;
char   *var_virt_dsn_filter;

 /*
  * Mappings.
  */
MAPS   *virtual_mailbox_maps;
MAPS   *virtual_uid_maps;
MAPS   *virtual_gid_maps;

 /*
  * Bit masks.
  */
int     virtual_mbox_lock_mask;

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
     */
    state.level = 0;
    deliver_attr_init(&state.msg_attr);
    state.msg_attr.queue_name = rqst->queue_name;
    state.msg_attr.queue_id = rqst->queue_id;
    state.msg_attr.fp = rqst->fp;
    state.msg_attr.offset = rqst->data_offset;
    state.msg_attr.sender = rqst->sender;
    state.msg_attr.dsn_envid = rqst->dsn_envid;
    state.msg_attr.dsn_ret = rqst->dsn_ret;
    state.msg_attr.relay = service;
    state.msg_attr.msg_stats = rqst->msg_stats;
    RESET_USER_ATTR(usr_attr, state.level);
    state.request = rqst;

    /*
     * Iterate over each recipient named in the delivery request. When the
     * mail delivery status for a given recipient is definite (i.e. bounced
     * or delivered), update the message queue file and cross off the
     * recipient. Update the per-message delivery status.
     */
    for (msg_stat = 0, rcpt = rqst->rcpt_list.info; rcpt < rcpt_end; rcpt++) {
	state.msg_attr.rcpt = *rcpt;
	rcpt_stat = deliver_recipient(state, usr_attr);
	if (rcpt_stat == 0 && (rqst->flags & DEL_REQ_FLAG_SUCCESS))
	    deliver_completed(state.msg_attr.fp, rcpt->offset);
	msg_stat |= rcpt_stat;
    }

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
     * Drop privileges most of the time.
     */
    set_eugid(var_owner_uid, var_owner_gid);

    /*
     * No case folding needed: the recipient address is case folded.
     */
    virtual_mailbox_maps =
	maps_create(VAR_VIRT_MAILBOX_MAPS, var_virt_mailbox_maps,
		    DICT_FLAG_LOCK | DICT_FLAG_PARANOID
		    | DICT_FLAG_UTF8_REQUEST);

    virtual_uid_maps =
	maps_create(VAR_VIRT_UID_MAPS, var_virt_uid_maps,
		    DICT_FLAG_LOCK | DICT_FLAG_PARANOID
		    | DICT_FLAG_UTF8_REQUEST);

    virtual_gid_maps =
	maps_create(VAR_VIRT_GID_MAPS, var_virt_gid_maps,
		    DICT_FLAG_LOCK | DICT_FLAG_PARANOID
		    | DICT_FLAG_UTF8_REQUEST);

    virtual_mbox_lock_mask = mbox_lock_mask(var_virt_mailbox_lock);
}

/* pre_init - pre-jail initialization */

static void pre_init(char *unused_name, char **unused_argv)
{

    /*
     * Reset the file size limit from the message size limit to the mailbox
     * size limit.
     * 
     * We can't have mailbox size limit smaller than the message size limit,
     * because that prohibits the delivery agent from updating the queue
     * file.
     */
    if (var_virt_mailbox_limit) {
	if (var_virt_mailbox_limit < var_message_limit || var_message_limit == 0)
	    msg_fatal("main.cf configuration error: %s is smaller than %s",
		      VAR_VIRT_MAILBOX_LIMIT, VAR_MESSAGE_LIMIT);
	set_file_limit(var_virt_mailbox_limit);
    }

    /*
     * flush client.
     */
    flush_init();
}

MAIL_VERSION_STAMP_DECLARE;

/* main - pass control to the single-threaded skeleton */

int     main(int argc, char **argv)
{
    static const CONFIG_INT_TABLE int_table[] = {
	VAR_VIRT_MINUID, DEF_VIRT_MINUID, &var_virt_minimum_uid, 1, 0,
	0,
    };
    static const CONFIG_LONG_TABLE long_table[] = {
	VAR_VIRT_MAILBOX_LIMIT, DEF_VIRT_MAILBOX_LIMIT, &var_virt_mailbox_limit, 0, 0,
	0,
    };
    static const CONFIG_STR_TABLE str_table[] = {
	VAR_MAIL_SPOOL_DIR, DEF_MAIL_SPOOL_DIR, &var_mail_spool_dir, 0, 0,
	VAR_VIRT_MAILBOX_MAPS, DEF_VIRT_MAILBOX_MAPS, &var_virt_mailbox_maps, 0, 0,
	VAR_VIRT_UID_MAPS, DEF_VIRT_UID_MAPS, &var_virt_uid_maps, 0, 0,
	VAR_VIRT_GID_MAPS, DEF_VIRT_GID_MAPS, &var_virt_gid_maps, 0, 0,
	VAR_VIRT_MAILBOX_BASE, DEF_VIRT_MAILBOX_BASE, &var_virt_mailbox_base, 1, 0,
	VAR_VIRT_MAILBOX_LOCK, DEF_VIRT_MAILBOX_LOCK, &var_virt_mailbox_lock, 1, 0,
	VAR_VIRT_DSN_FILTER, DEF_VIRT_DSN_FILTER, &var_virt_dsn_filter, 0, 0,
	0,
    };
    static const CONFIG_BOOL_TABLE bool_table[] = {
	VAR_STRICT_MBOX_OWNER, DEF_STRICT_MBOX_OWNER, &var_strict_mbox_owner,
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
		       CA_MAIL_SERVER_BOOL_TABLE(bool_table),
		       CA_MAIL_SERVER_PRE_INIT(pre_init),
		       CA_MAIL_SERVER_POST_INIT(post_init),
		       CA_MAIL_SERVER_PRE_ACCEPT(pre_accept),
		       CA_MAIL_SERVER_PRIVILEGED,
		       CA_MAIL_SERVER_BOUNCE_INIT(VAR_VIRT_DSN_FILTER,
						  &var_virt_dsn_filter),
		       0);
}
