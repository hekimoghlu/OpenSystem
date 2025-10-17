/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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
#include <pwd.h>
#include <grp.h>
#include <fcntl.h>
#include <ctype.h>

#ifdef STRCASECMP_IN_STRINGS_H
#include <strings.h>
#endif

/* Utility library. */

#include <msg.h>
#include <vstream.h>
#include <vstring.h>
#include <argv.h>
#include <htable.h>
#include <dict.h>
#include <iostuff.h>
#include <mymalloc.h>
#include <mac_parse.h>
#include <set_eugid.h>
#include <split_at.h>
#include <stringops.h>

/* Global library. */

#include <recipient_list.h>
#include <deliver_request.h>
#include <mail_params.h>
#include <mail_version.h>
#include <mail_conf.h>
#include <bounce.h>
#include <defer.h>
#include <deliver_completed.h>
#include <sent.h>
#include <pipe_command.h>
#include <mail_copy.h>
#include <mail_addr.h>
#include <canon_addr.h>
#include <split_addr.h>
#include <off_cvt.h>
#include <quote_822_local.h>
#include <flush_clnt.h>
#include <dsn_util.h>
#include <dsn_buf.h>
#include <sys_exits.h>
#include <delivered_hdr.h>
#include <fold_addr.h>
#include <mail_parm_split.h>

/* Single server skeleton. */

#include <mail_server.h>

/* Application-specific. */

 /*
  * The mini symbol table name and keys used for expanding macros in
  * command-line arguments.
  * 
  * XXX Update  the parse_callback() routine when something gets added here,
  * even when the macro is not recipient dependent.
  */
#define PIPE_DICT_TABLE		"pipe_command"	/* table name */
#define PIPE_DICT_NEXTHOP	"nexthop"	/* key */
#define PIPE_DICT_RCPT		"recipient"	/* key */
#define PIPE_DICT_ORIG_RCPT	"original_recipient"	/* key */
#define PIPE_DICT_SENDER	"sender"/* key */
#define PIPE_DICT_USER		"user"	/* key */
#define PIPE_DICT_EXTENSION	"extension"	/* key */
#define PIPE_DICT_MAILBOX	"mailbox"	/* key */
#define PIPE_DICT_DOMAIN	"domain"/* key */
#define PIPE_DICT_SIZE		"size"	/* key */
#define PIPE_DICT_CLIENT_ADDR	"client_address"	/* key */
#define PIPE_DICT_CLIENT_NAME	"client_hostname"	/* key */
#define PIPE_DICT_CLIENT_PORT	"client_port"	/* key */
#define PIPE_DICT_CLIENT_PROTO	"client_protocol"	/* key */
#define PIPE_DICT_CLIENT_HELO	"client_helo"	/* key */
#define PIPE_DICT_SASL_METHOD	"sasl_method"	/* key */
#define PIPE_DICT_SASL_USERNAME	"sasl_username"	/* key */
#define PIPE_DICT_SASL_SENDER	"sasl_sender"	/* key */
#define PIPE_DICT_QUEUE_ID	"queue_id"	/* key */

 /*
  * Flags used to pass back the type of special parameter found by
  * parse_callback.
  */
#define PIPE_FLAG_RCPT		(1<<0)
#define PIPE_FLAG_USER		(1<<1)
#define PIPE_FLAG_EXTENSION	(1<<2)
#define PIPE_FLAG_MAILBOX	(1<<3)
#define PIPE_FLAG_DOMAIN	(1<<4)
#define PIPE_FLAG_ORIG_RCPT	(1<<5)

 /*
  * Additional flags. These are colocated with mail_copy() flags. Allow some
  * space for extension of the mail_copy() interface.
  */
#define PIPE_OPT_FOLD_BASE	(16)
#define PIPE_OPT_FOLD_USER	(FOLD_ADDR_USER << PIPE_OPT_FOLD_BASE)
#define PIPE_OPT_FOLD_HOST	(FOLD_ADDR_HOST << PIPE_OPT_FOLD_BASE)
#define PIPE_OPT_QUOTE_LOCAL	(1 << (PIPE_OPT_FOLD_BASE + 2))
#define PIPE_OPT_FINAL_DELIVERY	(1 << (PIPE_OPT_FOLD_BASE + 3))

#define PIPE_OPT_FOLD_ALL	(FOLD_ADDR_ALL << PIPE_OPT_FOLD_BASE)
#define PIPE_OPT_FOLD_FLAGS(f) \
	(((f) & PIPE_OPT_FOLD_ALL) >> PIPE_OPT_FOLD_BASE)

 /*
  * Tunable parameters. Values are taken from the config file, after
  * prepending the service name to _name, and so on.
  */
int     var_command_maxtime;		/* You can now leave this here. */

 /*
  * Other main.cf parameters.
  */
char   *var_pipe_dsn_filter;

 /*
  * For convenience. Instead of passing around lists of parameters, bundle
  * them up in convenient structures.
  */

 /*
  * Structure for service-specific configuration parameters.
  */
typedef struct {
    int     time_limit;			/* per-service time limit */
} PIPE_PARAMS;

 /*
  * Structure for command-line parameters.
  */
typedef struct {
    char  **command;			/* argument vector */
    uid_t   uid;			/* command privileges */
    gid_t   gid;			/* command privileges */
    int     flags;			/* mail_copy() flags */
    char   *exec_dir;			/* working directory */
    char   *chroot_dir;			/* chroot directory */
    VSTRING *eol;			/* output record delimiter */
    VSTRING *null_sender;		/* null sender expansion */
    off_t   size_limit;			/* max size in bytes we will accept */
} PIPE_ATTR;

 /*
  * Structure for command-line parameter macro expansion.
  */
typedef struct {
    const char *service;		/* for warnings */
    int     expand_flag;		/* callback result */
} PIPE_STATE;

 /*
  * Silly little macros.
  */
#define STR	vstring_str

/* parse_callback - callback for mac_parse() */

static int parse_callback(int type, VSTRING *buf, void *context)
{
    PIPE_STATE *state = (PIPE_STATE *) context;
    struct cmd_flags {
	const char *name;
	int     flags;
    };
    static struct cmd_flags cmd_flags[] = {
	PIPE_DICT_NEXTHOP, 0,
	PIPE_DICT_RCPT, PIPE_FLAG_RCPT,
	PIPE_DICT_ORIG_RCPT, PIPE_FLAG_ORIG_RCPT,
	PIPE_DICT_SENDER, 0,
	PIPE_DICT_USER, PIPE_FLAG_USER,
	PIPE_DICT_EXTENSION, PIPE_FLAG_EXTENSION,
	PIPE_DICT_MAILBOX, PIPE_FLAG_MAILBOX,
	PIPE_DICT_DOMAIN, PIPE_FLAG_DOMAIN,
	PIPE_DICT_SIZE, 0,
	PIPE_DICT_CLIENT_ADDR, 0,
	PIPE_DICT_CLIENT_NAME, 0,
	PIPE_DICT_CLIENT_PORT, 0,
	PIPE_DICT_CLIENT_PROTO, 0,
	PIPE_DICT_CLIENT_HELO, 0,
	PIPE_DICT_SASL_METHOD, 0,
	PIPE_DICT_SASL_USERNAME, 0,
	PIPE_DICT_SASL_SENDER, 0,
	PIPE_DICT_QUEUE_ID, 0,
	0, 0,
    };
    struct cmd_flags *p;

    /*
     * See if this command-line argument references a special macro.
     */
    if (type == MAC_PARSE_VARNAME) {
	for (p = cmd_flags; /* see below */ ; p++) {
	    if (p->name == 0) {
		msg_warn("file %s/%s: service %s: unknown macro name: \"%s\"",
			 var_config_dir, MASTER_CONF_FILE,
			 state->service, vstring_str(buf));
		return (MAC_PARSE_ERROR);
	    } else if (strcmp(vstring_str(buf), p->name) == 0) {
		state->expand_flag |= p->flags;
		return (0);
	    }
	}
    }
    return (0);
}

/* morph_recipient - morph a recipient address */

static void morph_recipient(VSTRING *buf, const char *address, int flags)
{
    VSTRING *temp = vstring_alloc(100);

    /*
     * Quote the recipient address as appropriate.
     */
    if (flags & PIPE_OPT_QUOTE_LOCAL)
	quote_822_local(temp, address);
    else
	vstring_strcpy(temp, address);

    /*
     * Fold the recipient address as appropriate.
     */
    fold_addr(buf, STR(temp), PIPE_OPT_FOLD_FLAGS(flags));

    vstring_free(temp);
}

/* expand_argv - expand macros in the argument vector */

static ARGV *expand_argv(const char *service, char **argv,
			         RECIPIENT_LIST *rcpt_list, int flags)
{
    VSTRING *buf = vstring_alloc(100);
    ARGV   *result;
    char  **cpp;
    PIPE_STATE state;
    int     i;
    char   *ext;
    char   *dom;

    /*
     * This appears to be simple operation (replace $name by its expansion).
     * However, it becomes complex because a command-line argument that
     * references $recipient must expand to as many command-line arguments as
     * there are recipients (that's wat programs called by sendmail expect).
     * So we parse each command-line argument, and depending on what we find,
     * we either expand the argument just once, or we expand it once for each
     * recipient. In either case we end up parsing the command-line argument
     * twice. The amount of CPU time wasted will be negligible.
     * 
     * Note: we can't use recursive macro expansion here, because recursion
     * would screw up mail addresses that contain $ characters.
     */
#define NO	0
#define EARLY_RETURN(x) { argv_free(result); vstring_free(buf); return (x); }

    result = argv_alloc(1);
    for (cpp = argv; *cpp; cpp++) {
	state.service = service;
	state.expand_flag = 0;
	if (mac_parse(*cpp, parse_callback, (void *) &state) & MAC_PARSE_ERROR)
	    EARLY_RETURN(0);
	if (state.expand_flag == 0) {		/* no $recipient etc. */
	    argv_add(result, dict_eval(PIPE_DICT_TABLE, *cpp, NO), ARGV_END);
	} else {				/* contains $recipient etc. */
	    for (i = 0; i < rcpt_list->len; i++) {

		/*
		 * This argument contains $recipient.
		 */
		if (state.expand_flag & PIPE_FLAG_RCPT) {
		    morph_recipient(buf, rcpt_list->info[i].address, flags);
		    dict_update(PIPE_DICT_TABLE, PIPE_DICT_RCPT, STR(buf));
		}

		/*
		 * This argument contains $original_recipient.
		 */
		if (state.expand_flag & PIPE_FLAG_ORIG_RCPT) {
		    morph_recipient(buf, rcpt_list->info[i].orig_addr, flags);
		    dict_update(PIPE_DICT_TABLE, PIPE_DICT_ORIG_RCPT, STR(buf));
		}

		/*
		 * This argument contains $user. Extract the plain user name.
		 * Either anything to the left of the extension delimiter or,
		 * in absence of the latter, anything to the left of the
		 * rightmost @.
		 * 
		 * Beware: if the user name is blank (e.g. +user@host), the
		 * argument is suppressed. This is necessary to allow for
		 * cyrus bulletin-board (global mailbox) delivery. XXX But,
		 * skipping empty user parts will also prevent other
		 * expansions of this specific command-line argument.
		 */
		if (state.expand_flag & PIPE_FLAG_USER) {
		    morph_recipient(buf, rcpt_list->info[i].address,
				    flags & PIPE_OPT_FOLD_ALL);
		    if (split_at_right(STR(buf), '@') == 0)
			msg_warn("no @ in recipient address: %s",
				 rcpt_list->info[i].address);
		    if (*var_rcpt_delim)
			split_addr(STR(buf), var_rcpt_delim);
		    if (*STR(buf) == 0)
			continue;
		    dict_update(PIPE_DICT_TABLE, PIPE_DICT_USER, STR(buf));
		}

		/*
		 * This argument contains $extension. Extract the recipient
		 * extension: anything between the leftmost extension
		 * delimiter and the rightmost @. The extension may be blank.
		 */
		if (state.expand_flag & PIPE_FLAG_EXTENSION) {
		    morph_recipient(buf, rcpt_list->info[i].address,
				    flags & PIPE_OPT_FOLD_ALL);
		    if (split_at_right(STR(buf), '@') == 0)
			msg_warn("no @ in recipient address: %s",
				 rcpt_list->info[i].address);
		    if (*var_rcpt_delim == 0
			|| (ext = split_addr(STR(buf), var_rcpt_delim)) == 0)
			ext = "";		/* insert null arg */
		    dict_update(PIPE_DICT_TABLE, PIPE_DICT_EXTENSION, ext);
		}

		/*
		 * This argument contains $mailbox. Extract the mailbox name:
		 * anything to the left of the rightmost @.
		 */
		if (state.expand_flag & PIPE_FLAG_MAILBOX) {
		    morph_recipient(buf, rcpt_list->info[i].address,
				    flags & PIPE_OPT_FOLD_ALL);
		    if (split_at_right(STR(buf), '@') == 0)
			msg_warn("no @ in recipient address: %s",
				 rcpt_list->info[i].address);
		    dict_update(PIPE_DICT_TABLE, PIPE_DICT_MAILBOX, STR(buf));
		}

		/*
		 * This argument contains $domain. Extract the domain name:
		 * anything to the right of the rightmost @.
		 */
		if (state.expand_flag & PIPE_FLAG_DOMAIN) {
		    morph_recipient(buf, rcpt_list->info[i].address,
				    flags & PIPE_OPT_FOLD_ALL);
		    dom = split_at_right(STR(buf), '@');
		    if (dom == 0) {
			msg_warn("no @ in recipient address: %s",
				 rcpt_list->info[i].address);
			dom = "";		/* insert null arg */
		    }
		    dict_update(PIPE_DICT_TABLE, PIPE_DICT_DOMAIN, dom);
		}

		/*
		 * Done.
		 */
		argv_add(result, dict_eval(PIPE_DICT_TABLE, *cpp, NO), ARGV_END);
	    }
	}
    }
    argv_terminate(result);
    vstring_free(buf);
    return (result);
}

/* get_service_params - get service-name dependent config information */

static void get_service_params(PIPE_PARAMS *config, char *service)
{
    const char *myname = "get_service_params";

    /*
     * Figure out the command time limit for this transport.
     */
    config->time_limit =
	get_mail_conf_time2(service, _MAXTIME, var_command_maxtime, 's', 1, 0);

    /*
     * Give the poor tester a clue of what is going on.
     */
    if (msg_verbose)
	msg_info("%s: time_limit %d", myname, config->time_limit);
}

/* get_service_attr - get command-line attributes */

static void get_service_attr(PIPE_ATTR *attr, char **argv)
{
    const char *myname = "get_service_attr";
    struct passwd *pwd;
    struct group *grp;
    char   *user;			/* user name */
    char   *group;			/* group name */
    char   *size;			/* max message size */
    char   *cp;

    /*
     * Initialize.
     */
    user = 0;
    group = 0;
    attr->command = 0;
    attr->flags = 0;
    attr->exec_dir = 0;
    attr->chroot_dir = 0;
    attr->eol = vstring_strcpy(vstring_alloc(1), "\n");
    attr->null_sender = vstring_strcpy(vstring_alloc(1), MAIL_ADDR_MAIL_DAEMON);
    attr->size_limit = 0;

    /*
     * Iterate over the command-line attribute list.
     */
    for ( /* void */ ; *argv != 0; argv++) {

	/*
	 * flags=stuff
	 */
	if (strncasecmp("flags=", *argv, sizeof("flags=") - 1) == 0) {
	    for (cp = *argv + sizeof("flags=") - 1; *cp; cp++) {
		switch (*cp) {
		case 'B':
		    attr->flags |= MAIL_COPY_BLANK;
		    break;
		case 'D':
		    attr->flags |= MAIL_COPY_DELIVERED;
		    break;
		case 'F':
		    attr->flags |= MAIL_COPY_FROM;
		    break;
		case 'O':
		    attr->flags |= MAIL_COPY_ORIG_RCPT;
		    break;
		case 'R':
		    attr->flags |= MAIL_COPY_RETURN_PATH;
		    break;
		case 'X':
		    attr->flags |= PIPE_OPT_FINAL_DELIVERY;
		    break;
		case '.':
		    attr->flags |= MAIL_COPY_DOT;
		    break;
		case '>':
		    attr->flags |= MAIL_COPY_QUOTE;
		    break;
		case 'h':
		    attr->flags |= PIPE_OPT_FOLD_HOST;
		    break;
		case 'q':
		    attr->flags |= PIPE_OPT_QUOTE_LOCAL;
		    break;
		case 'u':
		    attr->flags |= PIPE_OPT_FOLD_USER;
		    break;
		default:
		    msg_fatal("unknown flag: %c (ignored)", *cp);
		    break;
		}
	    }
	}

	/*
	 * user=username[:groupname]
	 */
	else if (strncasecmp("user=", *argv, sizeof("user=") - 1) == 0) {
	    user = *argv + sizeof("user=") - 1;
	    if ((group = split_at(user, ':')) != 0)	/* XXX clobbers argv */
		if (*group == 0)
		    group = 0;
	    if ((pwd = getpwnam(user)) == 0)
		msg_fatal("%s: unknown username: %s", myname, user);
	    attr->uid = pwd->pw_uid;
	    if (group != 0) {
		if ((grp = getgrnam(group)) == 0)
		    msg_fatal("%s: unknown group: %s", myname, group);
		attr->gid = grp->gr_gid;
	    } else {
		attr->gid = pwd->pw_gid;
	    }
	}

	/*
	 * directory=string
	 */
	else if (strncasecmp("directory=", *argv, sizeof("directory=") - 1) == 0) {
	    attr->exec_dir = mystrdup(*argv + sizeof("directory=") - 1);
	}

	/*
	 * chroot=string
	 */
	else if (strncasecmp("chroot=", *argv, sizeof("chroot=") - 1) == 0) {
	    attr->chroot_dir = mystrdup(*argv + sizeof("chroot=") - 1);
	}

	/*
	 * eol=string
	 */
	else if (strncasecmp("eol=", *argv, sizeof("eol=") - 1) == 0) {
	    unescape(attr->eol, *argv + sizeof("eol=") - 1);
	}

	/*
	 * null_sender=string
	 */
	else if (strncasecmp("null_sender=", *argv, sizeof("null_sender=") - 1) == 0) {
	    vstring_strcpy(attr->null_sender, *argv + sizeof("null_sender=") - 1);
	}

	/*
	 * size=max_message_size (in bytes)
	 */
	else if (strncasecmp("size=", *argv, sizeof("size=") - 1) == 0) {
	    size = *argv + sizeof("size=") - 1;
	    if ((attr->size_limit = off_cvt_string(size)) < 0)
		msg_fatal("%s: bad size= value: %s", myname, size);
	}

	/*
	 * argv=command...
	 */
	else if (strncasecmp("argv=", *argv, sizeof("argv=") - 1) == 0) {
	    *argv += sizeof("argv=") - 1;	/* XXX clobbers argv */
	    attr->command = argv;
	    break;
	}

	/*
	 * Bad.
	 */
	else
	    msg_fatal("unknown attribute name: %s", *argv);
    }

    /*
     * Sanity checks. Verify that every member has an acceptable value.
     */
    if (user == 0)
	msg_fatal("missing user= command-line attribute");
    if (attr->command == 0)
	msg_fatal("missing argv= command-line attribute");
    if (attr->uid == 0)
	msg_fatal("user= command-line attribute specifies root privileges");
    if (attr->uid == var_owner_uid)
	msg_fatal("user= command-line attribute specifies mail system owner %s",
		  var_mail_owner);
    if (attr->gid == 0)
	msg_fatal("user= command-line attribute specifies privileged group id 0");
    if (attr->gid == var_owner_gid)
	msg_fatal("user= command-line attribute specifies mail system owner %s group id %ld",
		  var_mail_owner, (long) attr->gid);
    if (attr->gid == var_sgid_gid)
	msg_fatal("user= command-line attribute specifies mail system %s group id %ld",
		  var_sgid_group, (long) attr->gid);

    /*
     * Give the poor tester a clue of what is going on.
     */
    if (msg_verbose)
	msg_info("%s: uid %ld, gid %ld, flags %d, size %ld",
		 myname, (long) attr->uid, (long) attr->gid,
		 attr->flags, (long) attr->size_limit);
}

/* eval_command_status - do something with command completion status */

static int eval_command_status(int command_status, char *service,
			          DELIVER_REQUEST *request, PIPE_ATTR *attr,
			               DSN_BUF *why)
{
    RECIPIENT *rcpt;
    int     status;
    int     result = 0;
    int     n;
    char   *saved_text;

    /*
     * Depending on the result, bounce or defer the message, and mark the
     * recipient as done where appropriate.
     */
    switch (command_status) {
    case PIPE_STAT_OK:
	/* Save the command output before dsb_update() clobbers it. */
	vstring_truncate(why->reason, trimblanks(STR(why->reason),
			      VSTRING_LEN(why->reason)) - STR(why->reason));
	if (VSTRING_LEN(why->reason) > 0) {
	    VSTRING_TERMINATE(why->reason);
	    saved_text =
		vstring_export(vstring_sprintf(
				    vstring_alloc(VSTRING_LEN(why->reason)),
					    " (%.100s)", STR(why->reason)));
	} else
	    saved_text = mystrdup("");		/* uses shared R/O storage */
	dsb_update(why, "2.0.0", (attr->flags & PIPE_OPT_FINAL_DELIVERY) ?
		   "delivered" : "relayed", DSB_SKIP_RMTA, DSB_SKIP_REPLY,
		   "delivered via %s service%s", service, saved_text);
	myfree(saved_text);
	(void) DSN_FROM_DSN_BUF(why);
	for (n = 0; n < request->rcpt_list.len; n++) {
	    rcpt = request->rcpt_list.info + n;
	    status = sent(DEL_REQ_TRACE_FLAGS(request->flags),
			  request->queue_id, &request->msg_stats, rcpt,
			  service, &why->dsn);
	    if (status == 0 && (request->flags & DEL_REQ_FLAG_SUCCESS))
		deliver_completed(request->fp, rcpt->offset);
	    result |= status;
	}
	break;
    case PIPE_STAT_BOUNCE:
    case PIPE_STAT_DEFER:
	(void) DSN_FROM_DSN_BUF(why);
	for (n = 0; n < request->rcpt_list.len; n++) {
	    rcpt = request->rcpt_list.info + n;
	    /* XXX Maybe encapsulate this with ndr_append(). */
	    status = (STR(why->status)[0] != '4' ?
		      bounce_append : defer_append)
		(DEL_REQ_TRACE_FLAGS(request->flags),
		 request->queue_id,
		 &request->msg_stats, rcpt,
		 service, &why->dsn);
	    if (status == 0)
		deliver_completed(request->fp, rcpt->offset);
	    result |= status;
	}
	break;
    case PIPE_STAT_CORRUPT:
	/* XXX DSN should we send something? */
	result |= DEL_STAT_DEFER;
	break;
    default:
	msg_panic("eval_command_status: bad status %d", command_status);
	/* NOTREACHED */
    }

    return (result);
}

/* deliver_message - deliver message with extreme prejudice */

static int deliver_message(DELIVER_REQUEST *request, char *service, char **argv)
{
    const char *myname = "deliver_message";
    static PIPE_PARAMS conf;
    static PIPE_ATTR attr;
    RECIPIENT_LIST *rcpt_list = &request->rcpt_list;
    DSN_BUF *why = dsb_create();
    VSTRING *buf;
    ARGV   *expanded_argv = 0;
    int     deliver_status;
    int     command_status;
    ARGV   *export_env;
    const char *sender;

#define DELIVER_MSG_CLEANUP() { \
	dsb_free(why); \
	if (expanded_argv) argv_free(expanded_argv); \
    }

    if (msg_verbose)
	msg_info("%s: from <%s>", myname, request->sender);

    /*
     * Sanity checks. The get_service_params() and get_service_attr()
     * routines also do some sanity checks. Look up service attributes and
     * config information only once. This is safe since the information comes
     * from a trusted source, not from the delivery request.
     */
    if (request->nexthop[0] == 0)
	msg_fatal("empty nexthop hostname");
    if (rcpt_list->len <= 0)
	msg_fatal("recipient count: %d", rcpt_list->len);
    if (attr.command == 0) {
	get_service_params(&conf, service);
	get_service_attr(&attr, argv);
    }

    /*
     * The D flag cannot be specified for multi-recipient deliveries.
     */
    if ((attr.flags & MAIL_COPY_DELIVERED) && (rcpt_list->len > 1)) {
	dsb_simple(why, "4.3.5", "mail system configuration error");
	deliver_status = eval_command_status(PIPE_STAT_DEFER, service,
					     request, &attr, why);
	msg_warn("pipe flag `D' requires %s_destination_recipient_limit = 1",
		 service);
	DELIVER_MSG_CLEANUP();
	return (deliver_status);
    }

    /*
     * The O flag cannot be specified for multi-recipient deliveries.
     */
    if ((attr.flags & MAIL_COPY_ORIG_RCPT) && (rcpt_list->len > 1)) {
	dsb_simple(why, "4.3.5", "mail system configuration error");
	deliver_status = eval_command_status(PIPE_STAT_DEFER, service,
					     request, &attr, why);
	msg_warn("pipe flag `O' requires %s_destination_recipient_limit = 1",
		 service);
	DELIVER_MSG_CLEANUP();
	return (deliver_status);
    }

    /*
     * Check that this agent accepts messages this large.
     */
    if (attr.size_limit != 0 && request->data_size > attr.size_limit) {
	if (msg_verbose)
	    msg_info("%s: too big: size_limit = %ld, request->data_size = %ld",
		     myname, (long) attr.size_limit, request->data_size);
	dsb_simple(why, "5.2.3", "message too large");
	deliver_status = eval_command_status(PIPE_STAT_BOUNCE, service,
					     request, &attr, why);
	DELIVER_MSG_CLEANUP();
	return (deliver_status);
    }

    /*
     * Don't deliver a trace-only request.
     */
    if (DEL_REQ_TRACE_ONLY(request->flags)) {
	RECIPIENT *rcpt;
	int     status;
	int     n;

	deliver_status = 0;
	dsb_simple(why, "2.0.0", "delivers to command: %s", attr.command[0]);
	(void) DSN_FROM_DSN_BUF(why);
	for (n = 0; n < request->rcpt_list.len; n++) {
	    rcpt = request->rcpt_list.info + n;
	    status = sent(DEL_REQ_TRACE_FLAGS(request->flags),
			  request->queue_id, &request->msg_stats,
			  rcpt, service, &why->dsn);
	    if (status == 0 && (request->flags & DEL_REQ_FLAG_SUCCESS))
		deliver_completed(request->fp, rcpt->offset);
	    deliver_status |= status;
	}
	DELIVER_MSG_CLEANUP();
	return (deliver_status);
    }

    /*
     * Report mail delivery loops. By definition, this requires
     * single-recipient delivery. Don't silently lose recipients.
     */
    if (attr.flags & MAIL_COPY_DELIVERED) {
	DELIVERED_HDR_INFO *info;
	RECIPIENT *rcpt;
	int     loop_found;

	if (request->rcpt_list.len > 1)
	    msg_panic("%s: delivered-to enabled with multi-recipient request",
		      myname);
	info = delivered_hdr_init(request->fp, request->data_offset,
				  FOLD_ADDR_ALL);
	rcpt = request->rcpt_list.info;
	loop_found = delivered_hdr_find(info, rcpt->address);
	delivered_hdr_free(info);
	if (loop_found) {
	    dsb_simple(why, "5.4.6", "mail forwarding loop for %s",
		       rcpt->address);
	    deliver_status = eval_command_status(PIPE_STAT_BOUNCE, service,
						 request, &attr, why);
	    DELIVER_MSG_CLEANUP();
	    return (deliver_status);
	}
    }

    /*
     * Deliver. Set the nexthop and sender variables, and expand the command
     * argument vector. Recipients will be expanded on the fly. XXX Rewrite
     * envelope and header addresses according to transport-specific
     * rewriting rules.
     */
    if (vstream_fseek(request->fp, request->data_offset, SEEK_SET) < 0)
	msg_fatal("seek queue file %s: %m", VSTREAM_PATH(request->fp));

    /*
     * A non-empty null sender replacement is subject to the 'q' flag.
     */
    buf = vstring_alloc(10);
    sender = *request->sender ? request->sender : STR(attr.null_sender);
    if (*sender && (attr.flags & PIPE_OPT_QUOTE_LOCAL)) {
	quote_822_local(buf, sender);
	dict_update(PIPE_DICT_TABLE, PIPE_DICT_SENDER, STR(buf));
    } else
	dict_update(PIPE_DICT_TABLE, PIPE_DICT_SENDER, sender);
    if (attr.flags & PIPE_OPT_FOLD_HOST) {
	casefold(buf, request->nexthop);
	dict_update(PIPE_DICT_TABLE, PIPE_DICT_NEXTHOP, STR(buf));
    } else
	dict_update(PIPE_DICT_TABLE, PIPE_DICT_NEXTHOP, request->nexthop);
    vstring_sprintf(buf, "%ld", (long) request->data_size);
    dict_update(PIPE_DICT_TABLE, PIPE_DICT_SIZE, STR(buf));
    dict_update(PIPE_DICT_TABLE, PIPE_DICT_CLIENT_ADDR,
		request->client_addr);
    dict_update(PIPE_DICT_TABLE, PIPE_DICT_CLIENT_HELO,
		request->client_helo);
    dict_update(PIPE_DICT_TABLE, PIPE_DICT_CLIENT_NAME,
		request->client_name);
    dict_update(PIPE_DICT_TABLE, PIPE_DICT_CLIENT_PORT,
		request->client_port);
    dict_update(PIPE_DICT_TABLE, PIPE_DICT_CLIENT_PROTO,
		request->client_proto);
    dict_update(PIPE_DICT_TABLE, PIPE_DICT_SASL_METHOD,
		request->sasl_method);
    dict_update(PIPE_DICT_TABLE, PIPE_DICT_SASL_USERNAME,
		request->sasl_username);
    dict_update(PIPE_DICT_TABLE, PIPE_DICT_SASL_SENDER,
		request->sasl_sender);
    dict_update(PIPE_DICT_TABLE, PIPE_DICT_QUEUE_ID,
		request->queue_id);
    vstring_free(buf);

    if ((expanded_argv = expand_argv(service, attr.command,
				     rcpt_list, attr.flags)) == 0) {
	dsb_simple(why, "4.3.5", "mail system configuration error");
	deliver_status = eval_command_status(PIPE_STAT_DEFER, service,
					     request, &attr, why);
	DELIVER_MSG_CLEANUP();
	return (deliver_status);
    }
    export_env = mail_parm_split(VAR_EXPORT_ENVIRON, var_export_environ);

    command_status = pipe_command(request->fp, why,
				  CA_PIPE_CMD_UID(attr.uid),
				  CA_PIPE_CMD_GID(attr.gid),
				  CA_PIPE_CMD_SENDER(sender),
				  CA_PIPE_CMD_COPY_FLAGS(attr.flags),
				  CA_PIPE_CMD_ARGV(expanded_argv->argv),
				  CA_PIPE_CMD_TIME_LIMIT(conf.time_limit),
				  CA_PIPE_CMD_EOL(STR(attr.eol)),
				  CA_PIPE_CMD_EXPORT(export_env->argv),
				  CA_PIPE_CMD_CWD(attr.exec_dir),
				  CA_PIPE_CMD_CHROOT(attr.chroot_dir),
			CA_PIPE_CMD_ORIG_RCPT(rcpt_list->info[0].orig_addr),
			  CA_PIPE_CMD_DELIVERED(rcpt_list->info[0].address),
				  CA_PIPE_CMD_END);
    argv_free(export_env);

    deliver_status = eval_command_status(command_status, service, request,
					 &attr, why);

    /*
     * Clean up.
     */
    DELIVER_MSG_CLEANUP();

    return (deliver_status);
}

/* pipe_service - perform service for client */

static void pipe_service(VSTREAM *client_stream, char *service, char **argv)
{
    DELIVER_REQUEST *request;
    int     status;

    /*
     * This routine runs whenever a client connects to the UNIX-domain socket
     * dedicated to delivery via external command. What we see below is a
     * little protocol to (1) tell the queue manager that we are ready, (2)
     * read a request from the queue manager, and (3) report the completion
     * status of that request. All connection-management stuff is handled by
     * the common code in single_server.c.
     */
    if ((request = deliver_request_read(client_stream)) != 0) {
	status = deliver_message(request, service, argv);
	deliver_request_done(client_stream, request, status);
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

/* drop_privileges - drop privileges most of the time */

static void drop_privileges(char *unused_name, char **unused_argv)
{
    set_eugid(var_owner_uid, var_owner_gid);
}

/* pre_init - initialize */

static void pre_init(char *unused_name, char **unused_argv)
{
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
    static const CONFIG_STR_TABLE str_table[] = {
	VAR_PIPE_DSN_FILTER, DEF_PIPE_DSN_FILTER, &var_pipe_dsn_filter, 0, 0,
	0,
    };

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    single_server_main(argc, argv, pipe_service,
		       CA_MAIL_SERVER_TIME_TABLE(time_table),
		       CA_MAIL_SERVER_STR_TABLE(str_table),
		       CA_MAIL_SERVER_PRE_INIT(pre_init),
		       CA_MAIL_SERVER_POST_INIT(drop_privileges),
		       CA_MAIL_SERVER_PRE_ACCEPT(pre_accept),
		       CA_MAIL_SERVER_PRIVILEGED,
		       CA_MAIL_SERVER_BOUNCE_INIT(VAR_PIPE_DSN_FILTER,
						  &var_pipe_dsn_filter),
		       0);
}
