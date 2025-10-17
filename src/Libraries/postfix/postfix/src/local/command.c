/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 4, 2023.
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
#include <stdarg.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <htable.h>
#include <vstring.h>
#include <vstream.h>
#include <argv.h>
#include <mac_parse.h>

/* Global library. */

#include <defer.h>
#include <bounce.h>
#include <sent.h>
#include <been_here.h>
#include <mail_params.h>
#include <pipe_command.h>
#include <mail_copy.h>
#include <dsn_util.h>
#include <mail_parm_split.h>

/* Application-specific. */

#include "local.h"

/* deliver_command - deliver to shell command */

int     deliver_command(LOCAL_STATE state, USER_ATTR usr_attr, const char *command)
{
    const char *myname = "deliver_command";
    DSN_BUF *why = state.msg_attr.why;
    int     cmd_status;
    int     deliver_status;
    ARGV   *env;
    int     copy_flags;
    char  **cpp;
    char   *cp;
    ARGV   *export_env;
    VSTRING *exec_dir;
    int     expand_status;

    /*
     * Make verbose logging easier to understand.
     */
    state.level++;
    if (msg_verbose)
	MSG_LOG_STATE(myname, state);

    /*
     * DUPLICATE ELIMINATION
     * 
     * Skip this command if it was already delivered to as this user.
     */
    if (been_here(state.dup_filter, "command %s:%ld %s",
		  state.msg_attr.user, (long) usr_attr.uid, command))
	return (0);

    /*
     * Don't deliver a trace-only request.
     */
    if (DEL_REQ_TRACE_ONLY(state.request->flags)) {
	dsb_simple(why, "2.0.0", "delivers to command: %s", command);
	return (sent(BOUNCE_FLAGS(state.request),
		     SENT_ATTR(state.msg_attr)));
    }

    /*
     * DELIVERY RIGHTS
     * 
     * Choose a default uid and gid when none have been selected (i.e. values
     * are still zero).
     */
    if (usr_attr.uid == 0 && (usr_attr.uid = var_default_uid) == 0)
	msg_panic("privileged default user id");
    if (usr_attr.gid == 0 && (usr_attr.gid = var_default_gid) == 0)
	msg_panic("privileged default group id");

    /*
     * Deliver.
     */
    copy_flags = MAIL_COPY_FROM | MAIL_COPY_RETURN_PATH
	| MAIL_COPY_ORIG_RCPT;
    if (local_deliver_hdr_mask & DELIVER_HDR_CMD)
	copy_flags |= MAIL_COPY_DELIVERED;

    if (vstream_fseek(state.msg_attr.fp, state.msg_attr.offset, SEEK_SET) < 0)
	msg_fatal("%s: seek queue file %s: %m",
		  myname, VSTREAM_PATH(state.msg_attr.fp));

    /*
     * Pass additional environment information. XXX This should be
     * configurable. However, passing untrusted information via environment
     * parameters opens up a whole can of worms. Lesson from web servers:
     * don't let any network data even near a shell. It causes trouble.
     */
    env = argv_alloc(1);
    if (usr_attr.home)
	argv_add(env, "HOME", usr_attr.home, ARGV_END);
    argv_add(env,
	     "LOGNAME", state.msg_attr.user,
	     "USER", state.msg_attr.user,
	     "SENDER", state.msg_attr.sender,
	     "RECIPIENT", state.msg_attr.rcpt.address,
	     "LOCAL", state.msg_attr.local,
	     ARGV_END);
    if (usr_attr.shell)
	argv_add(env, "SHELL", usr_attr.shell, ARGV_END);
    if (state.msg_attr.domain)
	argv_add(env, "DOMAIN", state.msg_attr.domain, ARGV_END);
    if (state.msg_attr.extension)
	argv_add(env, "EXTENSION", state.msg_attr.extension, ARGV_END);
    if (state.msg_attr.rcpt.orig_addr && state.msg_attr.rcpt.orig_addr[0])
	argv_add(env, "ORIGINAL_RECIPIENT", state.msg_attr.rcpt.orig_addr,
		 ARGV_END);

#define EXPORT_REQUEST(name, value) \
	if ((value)[0]) argv_add(env, (name), (value), ARGV_END);

    EXPORT_REQUEST("CLIENT_HOSTNAME", state.msg_attr.request->client_name);
    EXPORT_REQUEST("CLIENT_ADDRESS", state.msg_attr.request->client_addr);
    EXPORT_REQUEST("CLIENT_HELO", state.msg_attr.request->client_helo);
    EXPORT_REQUEST("CLIENT_PROTOCOL", state.msg_attr.request->client_proto);
    EXPORT_REQUEST("SASL_METHOD", state.msg_attr.request->sasl_method);
    EXPORT_REQUEST("SASL_SENDER", state.msg_attr.request->sasl_sender);
    EXPORT_REQUEST("SASL_USERNAME", state.msg_attr.request->sasl_username);

    argv_terminate(env);

    /*
     * Censor out undesirable characters from exported data.
     */
    for (cpp = env->argv; *cpp; cpp += 2)
	for (cp = cpp[1]; *(cp += strspn(cp, var_cmd_exp_filter)) != 0;)
	    *cp++ = '_';

    /*
     * Evaluate the command execution directory. Defer delivery if expansion
     * fails.
     */
    export_env = mail_parm_split(VAR_EXPORT_ENVIRON, var_export_environ);
    exec_dir = vstring_alloc(10);
    expand_status = local_expand(exec_dir, var_exec_directory,
				 &state, &usr_attr, var_exec_exp_filter);

    if (expand_status & MAC_PARSE_ERROR) {
	cmd_status = PIPE_STAT_DEFER;
	dsb_simple(why, "4.3.5", "mail system configuration error");
	msg_warn("bad parameter value syntax for %s: %s",
		 VAR_EXEC_DIRECTORY, var_exec_directory);
    } else {
	cmd_status = pipe_command(state.msg_attr.fp, why,
				  CA_PIPE_CMD_UID(usr_attr.uid),
				  CA_PIPE_CMD_GID(usr_attr.gid),
				  CA_PIPE_CMD_COMMAND(command),
				  CA_PIPE_CMD_COPY_FLAGS(copy_flags),
				  CA_PIPE_CMD_SENDER(state.msg_attr.sender),
		       CA_PIPE_CMD_ORIG_RCPT(state.msg_attr.rcpt.orig_addr),
			    CA_PIPE_CMD_DELIVERED(state.msg_attr.delivered),
				CA_PIPE_CMD_TIME_LIMIT(var_command_maxtime),
				  CA_PIPE_CMD_ENV(env->argv),
				  CA_PIPE_CMD_EXPORT(export_env->argv),
				  CA_PIPE_CMD_SHELL(var_local_cmd_shell),
				  CA_PIPE_CMD_CWD(*STR(exec_dir) ?
						STR(exec_dir) : (char *) 0),
				  CA_PIPE_CMD_END);
    }
    vstring_free(exec_dir);
    argv_free(export_env);
    argv_free(env);

    /*
     * Depending on the result, bounce or defer the message.
     */
    switch (cmd_status) {
    case PIPE_STAT_OK:
	dsb_simple(why, "2.0.0", "delivered to command: %s", command);
	deliver_status = sent(BOUNCE_FLAGS(state.request),
			      SENT_ATTR(state.msg_attr));
	break;
    case PIPE_STAT_BOUNCE:
    case PIPE_STAT_DEFER:
	/* Account for possible owner- sender address override. */
	deliver_status = bounce_workaround(state);
	break;
    case PIPE_STAT_CORRUPT:
	deliver_status = DEL_STAT_DEFER;
	break;
    default:
	msg_panic("%s: bad status %d", myname, cmd_status);
	/* NOTREACHED */
    }

    return (deliver_status);
}
