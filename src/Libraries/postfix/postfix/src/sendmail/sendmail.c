/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 7, 2021.
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
#include <unistd.h>
#include <string.h>
#include <stdio.h>			/* remove() */
#include <stdlib.h>
#include <signal.h>
#include <fcntl.h>
#include <syslog.h>
#include <time.h>
#include <errno.h>
#include <ctype.h>
#include <stdarg.h>
#include <sysexits.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <vstream.h>
#include <msg_vstream.h>
#include <msg_syslog.h>
#include <vstring_vstream.h>
#include <username.h>
#include <fullname.h>
#include <argv.h>
#include <safe.h>
#include <iostuff.h>
#include <stringops.h>
#include <set_ugid.h>
#include <connect.h>
#include <split_at.h>
#include <name_code.h>
#include <warn_stat.h>
#include <clean_env.h>

/* Global library. */

#include <mail_queue.h>
#include <mail_proto.h>
#include <mail_params.h>
#include <mail_version.h>
#include <record.h>
#include <rec_type.h>
#include <rec_streamlf.h>
#include <mail_conf.h>
#include <cleanup_user.h>
#include <mail_task.h>
#include <mail_run.h>
#include <debug_process.h>
#include <tok822.h>
#include <mail_flush.h>
#include <mail_stream.h>
#include <verp_sender.h>
#include <deliver_request.h>
#include <mime_state.h>
#include <header_opts.h>
#include <user_acl.h>
#include <dsn_mask.h>
#include <mail_parm_split.h>

/* Application-specific. */

 /*
  * Modes of operation.
  */
#define SM_MODE_ENQUEUE		1	/* delivery mode */
#define SM_MODE_NEWALIAS	2	/* initialize alias database */
#define SM_MODE_MAILQ		3	/* list mail queue */
#define SM_MODE_DAEMON		4	/* daemon mode */
#define SM_MODE_USER		5	/* user (stand-alone) mode */
#define SM_MODE_FLUSHQ		6	/* user (stand-alone) mode */
#define SM_MODE_IGNORE		7	/* ignore this mode */

 /*
  * Flag parade. Flags 8-15 are reserved for delivery request trace flags.
  */
#define SM_FLAG_AEOF	(1<<0)		/* archaic EOF */
#define SM_FLAG_XRCPT	(1<<1)		/* extract recipients from headers */

#define SM_FLAG_DEFAULT	(SM_FLAG_AEOF)

 /*
  * VERP support.
  */
static char *verp_delims;

 /*
  * Callback context for extracting recipients.
  */
typedef struct SM_STATE {
    VSTREAM *dst;			/* output stream */
    ARGV   *recipients;			/* recipients from regular headers */
    ARGV   *resent_recip;		/* recipients from resent headers */
    int     resent;			/* resent flag */
    const char *saved_sender;		/* for error messages */
    uid_t   uid;			/* for error messages */
    VSTRING *temp;			/* scratch buffer */
} SM_STATE;

 /*
  * Mail submission ACL, line-end fixing.
  */
char   *var_submit_acl;
char   *var_sm_fix_eol;

static const CONFIG_STR_TABLE str_table[] = {
    VAR_SUBMIT_ACL, DEF_SUBMIT_ACL, &var_submit_acl, 0, 0,
    VAR_SM_FIX_EOL, DEF_SM_FIX_EOL, &var_sm_fix_eol, 1, 0,
    0,
};

 /*
  * Silly little macros (SLMs).
  */
#define STR	vstring_str

/* output_text - output partial or complete text line */

static void output_text(void *context, int rec_type, const char *buf, ssize_t len,
			        off_t unused_offset)
{
    SM_STATE *state = (SM_STATE *) context;

    if (rec_put(state->dst, rec_type, buf, len) < 0)
	msg_fatal_status(EX_TEMPFAIL,
			 "%s(%ld): error writing queue file: %m",
			 state->saved_sender, (long) state->uid);
}

/* output_header - output one message header */

static void output_header(void *context, int header_class,
			          const HEADER_OPTS *header_info,
			          VSTRING *buf, off_t offset)
{
    SM_STATE *state = (SM_STATE *) context;
    TOK822 *tree;
    TOK822 **addr_list;
    TOK822 **tpp;
    ARGV   *rcpt;
    char   *start;
    char   *line;
    char   *next_line;
    ssize_t len;

    /*
     * Parse the header line, and save copies of recipient addresses in the
     * appropriate place.
     */
    if (header_class == MIME_HDR_PRIMARY
	&& header_info
	&& (header_info->flags & HDR_OPT_RECIP)
	&& (header_info->flags & HDR_OPT_EXTRACT)
	&& (state->resent == 0 || (header_info->flags & HDR_OPT_RR))) {
	if (header_info->flags & HDR_OPT_RR) {
	    rcpt = state->resent_recip;
	    if (state->resent == 0)
		state->resent = 1;
	} else
	    rcpt = state->recipients;
	tree = tok822_parse(STR(buf) + strlen(header_info->name) + 1);
	addr_list = tok822_grep(tree, TOK822_ADDR);
	for (tpp = addr_list; *tpp; tpp++) {
	    tok822_internalize(state->temp, tpp[0]->head, TOK822_STR_DEFL);
	    argv_add(rcpt, STR(state->temp), (char *) 0);
	}
	myfree((void *) addr_list);
	tok822_free_tree(tree);
    }

    /*
     * Pipe the unmodified message header through the header line folding
     * routine, and ensure that long lines are chopped appropriately.
     */
    for (line = start = STR(buf); line; line = next_line) {
	next_line = split_at(line, '\n');
	len = next_line ? next_line - line - 1 : strlen(line);
	do {
	    if (len > var_line_limit) {
		output_text(context, REC_TYPE_CONT, line, var_line_limit, offset);
		line += var_line_limit;
		len -= var_line_limit;
		offset += var_line_limit;
	    } else {
		output_text(context, REC_TYPE_NORM, line, len, offset);
		offset += len;
		break;
	    }
	} while (len > 0);
	offset += 1;
    }
}

/* enqueue - post one message */

static void enqueue(const int flags, const char *encoding,
		         const char *dsn_envid, int dsn_ret, int dsn_notify,
		            const char *rewrite_context, const char *sender,
		            const char *full_name, char **recipients)
{
    VSTRING *buf;
    VSTREAM *dst;
    char   *saved_sender;
    char  **cpp;
    int     type;
    char   *start;
    int     skip_from_;
    TOK822 *tree;
    TOK822 *tp;
    int     rcpt_count = 0;
    enum {
	STRIP_CR_DUNNO, STRIP_CR_DO, STRIP_CR_DONT, STRIP_CR_ERROR
    }       strip_cr;
    MAIL_STREAM *handle;
    VSTRING *postdrop_command;
    uid_t   uid = getuid();
    int     status;
    int     naddr;
    int     prev_type;
    MIME_STATE *mime_state = 0;
    SM_STATE state;
    int     mime_errs;
    const char *errstr;
    int     addr_count;
    int     level;
    static NAME_CODE sm_fix_eol_table[] = {
	SM_FIX_EOL_ALWAYS, STRIP_CR_DO,
	SM_FIX_EOL_STRICT, STRIP_CR_DUNNO,
	SM_FIX_EOL_NEVER, STRIP_CR_DONT,
	0, STRIP_CR_ERROR,
    };

    /*
     * Access control is enforced in the postdrop command. The code here
     * merely produces a more user-friendly interface.
     */
    if ((errstr = check_user_acl_byuid(VAR_SUBMIT_ACL,
				       var_submit_acl, uid)) != 0)
	msg_fatal_status(EX_NOPERM,
	  "User %s(%ld) is not allowed to submit mail", errstr, (long) uid);

    /*
     * Initialize.
     */
    buf = vstring_alloc(100);

    /*
     * Stop run-away process accidents by limiting the queue file size. This
     * is not a defense against DOS attack.
     */
    if (var_message_limit > 0 && get_file_limit() > var_message_limit)
	set_file_limit((off_t) var_message_limit);

    /*
     * The sender name is provided by the user. In principle, the mail pickup
     * service could deduce the sender name from queue file ownership, but:
     * pickup would not be able to run chrooted, and it may not be desirable
     * to use login names at all.
     */
    if (sender != 0) {
	VSTRING_RESET(buf);
	VSTRING_TERMINATE(buf);
	tree = tok822_parse(sender);
	for (naddr = 0, tp = tree; tp != 0; tp = tp->next)
	    if (tp->type == TOK822_ADDR && naddr++ == 0)
		tok822_internalize(buf, tp->head, TOK822_STR_DEFL);
	tok822_free_tree(tree);
	saved_sender = mystrdup(STR(buf));
	if (naddr > 1)
	    msg_warn("-f option specified malformed sender: %s", sender);
    } else {
	if ((sender = username()) == 0)
	    msg_fatal_status(EX_OSERR, "no login name found for user ID %lu",
			     (unsigned long) uid);
	saved_sender = mystrdup(sender);
    }

    /*
     * Let the postdrop command open the queue file for us, and sanity check
     * the content. XXX Make postdrop a manifest constant.
     */
    errno = 0;
    postdrop_command = vstring_alloc(1000);
    vstring_sprintf(postdrop_command, "%s/postdrop -r", var_command_dir);
    for (level = 0; level < msg_verbose; level++)
	vstring_strcat(postdrop_command, " -v");
    if ((handle = mail_stream_command(STR(postdrop_command))) == 0)
	msg_fatal_status(EX_UNAVAILABLE, "%s(%ld): unable to execute %s: %m",
			 saved_sender, (long) uid, STR(postdrop_command));
    vstring_free(postdrop_command);
    dst = handle->stream;

    /*
     * First, write envelope information to the output stream.
     * 
     * For sendmail compatibility, parse each command-line recipient as if it
     * were an RFC 822 message header; some MUAs specify comma-separated
     * recipient lists; and some MUAs even specify "word word <address>".
     * 
     * Sort-uniq-ing the recipient list is done after address canonicalization,
     * before recipients are written to queue file. That's cleaner than
     * having the queue manager nuke duplicate recipient status records.
     * 
     * XXX Should limit the size of envelope records.
     * 
     * With "sendmail -N", instead of a per-message NOTIFY record we store one
     * per recipient so that we can simplify the implementation somewhat.
     */
    if (dsn_envid)
	rec_fprintf(dst, REC_TYPE_ATTR, "%s=%s",
		    MAIL_ATTR_DSN_ENVID, dsn_envid);
    if (dsn_ret)
	rec_fprintf(dst, REC_TYPE_ATTR, "%s=%d",
		    MAIL_ATTR_DSN_RET, dsn_ret);
    rec_fprintf(dst, REC_TYPE_ATTR, "%s=%s",
		MAIL_ATTR_RWR_CONTEXT, rewrite_context);
    if (full_name || (full_name = fullname()) != 0)
	rec_fputs(dst, REC_TYPE_FULL, full_name);
    rec_fputs(dst, REC_TYPE_FROM, saved_sender);
    if (verp_delims && *saved_sender == 0)
	msg_fatal_status(EX_USAGE,
		      "%s(%ld): -V option requires non-null sender address",
			 saved_sender, (long) uid);
    if (encoding)
	rec_fprintf(dst, REC_TYPE_ATTR, "%s=%s", MAIL_ATTR_ENCODING, encoding);
    if (DEL_REQ_TRACE_FLAGS(flags))
	rec_fprintf(dst, REC_TYPE_ATTR, "%s=%d", MAIL_ATTR_TRACE_FLAGS,
		    DEL_REQ_TRACE_FLAGS(flags));
    if (verp_delims)
	rec_fputs(dst, REC_TYPE_VERP, verp_delims);
    if (recipients) {
	for (cpp = recipients; *cpp != 0; cpp++) {
	    tree = tok822_parse(*cpp);
	    for (addr_count = 0, tp = tree; tp != 0; tp = tp->next) {
		if (tp->type == TOK822_ADDR) {
		    tok822_internalize(buf, tp->head, TOK822_STR_DEFL);
		    if (dsn_notify)
			rec_fprintf(dst, REC_TYPE_ATTR, "%s=%d",
				    MAIL_ATTR_DSN_NOTIFY, dsn_notify);
		    if (REC_PUT_BUF(dst, REC_TYPE_RCPT, buf) < 0)
			msg_fatal_status(EX_TEMPFAIL,
				    "%s(%ld): error writing queue file: %m",
					 saved_sender, (long) uid);
		    ++rcpt_count;
		    ++addr_count;
		}
	    }
	    tok822_free_tree(tree);
	    if (addr_count == 0) {
		if (rec_put(dst, REC_TYPE_RCPT, "", 0) < 0)
		    msg_fatal_status(EX_TEMPFAIL,
				     "%s(%ld): error writing queue file: %m",
				     saved_sender, (long) uid);
		++rcpt_count;
	    }
	}
    }

    /*
     * Append the message contents to the queue file. Write chunks of at most
     * 1kbyte. Internally, we use different record types for data ending in
     * LF and for data that doesn't, so we can actually be binary transparent
     * for local mail. Unfortunately, SMTP has no record continuation
     * convention, so there is no guarantee that arbitrary data will be
     * delivered intact via SMTP. Strip leading From_ lines. For the benefit
     * of UUCP environments, also get rid of leading >>>From_ lines.
     */
    rec_fputs(dst, REC_TYPE_MESG, "");
    if (DEL_REQ_TRACE_ONLY(flags) != 0) {
	if (flags & SM_FLAG_XRCPT)
	    msg_fatal_status(EX_USAGE, "%s(%ld): -t option cannot be used with -bv",
			     saved_sender, (long) uid);
	if (*saved_sender)
	    rec_fprintf(dst, REC_TYPE_NORM, "From: %s", saved_sender);
	rec_fprintf(dst, REC_TYPE_NORM, "Subject: probe");
	if (recipients) {
	    rec_fprintf(dst, REC_TYPE_CONT, "To:");
	    for (cpp = recipients; *cpp != 0; cpp++) {
		rec_fprintf(dst, REC_TYPE_NORM, "	%s%s",
			    *cpp, cpp[1] ? "," : "");
	    }
	}
    } else {

	/*
	 * Initialize the MIME processor and set up the callback context.
	 */
	if (flags & SM_FLAG_XRCPT) {
	    state.dst = dst;
	    state.recipients = argv_alloc(2);
	    state.resent_recip = argv_alloc(2);
	    state.resent = 0;
	    state.saved_sender = saved_sender;
	    state.uid = uid;
	    state.temp = vstring_alloc(10);
	    mime_state = mime_state_alloc(MIME_OPT_DISABLE_MIME
					  | MIME_OPT_REPORT_TRUNC_HEADER,
					  output_header,
					  (MIME_STATE_ANY_END) 0,
					  output_text,
					  (MIME_STATE_ANY_END) 0,
					  (MIME_STATE_ERR_PRINT) 0,
					  (void *) &state);
	}

	/*
	 * Process header/body lines.
	 */
	skip_from_ = 1;
	strip_cr = name_code(sm_fix_eol_table, NAME_CODE_FLAG_STRICT_CASE,
			     var_sm_fix_eol);
	if (strip_cr == STRIP_CR_ERROR)
	    msg_fatal_status(EX_USAGE,
		    "invalid %s value: %s", VAR_SM_FIX_EOL, var_sm_fix_eol);
	for (prev_type = 0; (type = rec_streamlf_get(VSTREAM_IN, buf, var_line_limit))
	     != REC_TYPE_EOF; prev_type = type) {
	    if (strip_cr == STRIP_CR_DUNNO && type == REC_TYPE_NORM) {
		if (VSTRING_LEN(buf) > 0 && vstring_end(buf)[-1] == '\r')
		    strip_cr = STRIP_CR_DO;
		else
		    strip_cr = STRIP_CR_DONT;
	    }
	    if (skip_from_) {
		if (type == REC_TYPE_NORM) {
		    start = STR(buf);
		    if (strncmp(start + strspn(start, ">"), "From ", 5) == 0)
			continue;
		}
		skip_from_ = 0;
	    }
	    if (strip_cr == STRIP_CR_DO && type == REC_TYPE_NORM)
		while (VSTRING_LEN(buf) > 0 && vstring_end(buf)[-1] == '\r')
		    vstring_truncate(buf, VSTRING_LEN(buf) - 1);
	    if ((flags & SM_FLAG_AEOF) && prev_type != REC_TYPE_CONT
		&& VSTRING_LEN(buf) == 1 && *STR(buf) == '.')
		break;
	    if (mime_state) {
		mime_errs = mime_state_update(mime_state, type, STR(buf),
					      VSTRING_LEN(buf));
		if (mime_errs)
		    msg_fatal_status(EX_DATAERR,
				"%s(%ld): unable to extract recipients: %s",
				     saved_sender, (long) uid,
				     mime_state_error(mime_errs));
	    } else {
		if (REC_PUT_BUF(dst, type, buf) < 0)
		    msg_fatal_status(EX_TEMPFAIL,
				     "%s(%ld): error writing queue file: %m",
				     saved_sender, (long) uid);
	    }
	}
    }

    /*
     * Finish MIME processing. We need a final mime_state_update() call in
     * order to flush text that is still buffered. That can happen when the
     * last line did not end in newline.
     */
    if (mime_state) {
	mime_errs = mime_state_update(mime_state, REC_TYPE_EOF, "", 0);
	if (mime_errs)
	    msg_fatal_status(EX_DATAERR,
			     "%s(%ld): unable to extract recipients: %s",
			     saved_sender, (long) uid,
			     mime_state_error(mime_errs));
	mime_state = mime_state_free(mime_state);
    }

    /*
     * Append recipient addresses that were extracted from message headers.
     */
    rec_fputs(dst, REC_TYPE_XTRA, "");
    if (flags & SM_FLAG_XRCPT) {
	for (cpp = state.resent ? state.resent_recip->argv :
	     state.recipients->argv; *cpp; cpp++) {
	    if (dsn_notify)
		rec_fprintf(dst, REC_TYPE_ATTR, "%s=%d",
			    MAIL_ATTR_DSN_NOTIFY, dsn_notify);
	    if (rec_put(dst, REC_TYPE_RCPT, *cpp, strlen(*cpp)) < 0)
		msg_fatal_status(EX_TEMPFAIL,
				 "%s(%ld): error writing queue file: %m",
				 saved_sender, (long) uid);
	    ++rcpt_count;
	}
	argv_free(state.recipients);
	argv_free(state.resent_recip);
	vstring_free(state.temp);
    }
    if (rcpt_count == 0)
	msg_fatal_status(EX_USAGE, (flags & SM_FLAG_XRCPT) ?
		 "%s(%ld): No recipient addresses found in message header" :
			 "%s(%ld): Recipient addresses must be specified on"
			 " the command line or via the -t option",
			 saved_sender, (long) uid);

    /*
     * Identify the end of the queue file.
     */
    rec_fputs(dst, REC_TYPE_END, "");

    /*
     * Make sure that the message makes it to the file system. Once we have
     * terminated with successful exit status we cannot lose the message due
     * to "frivolous reasons". If all goes well, prevent the run-time error
     * handler from removing the file.
     */
    if (vstream_ferror(VSTREAM_IN))
	msg_fatal_status(EX_DATAERR, "%s(%ld): error reading input: %m",
			 saved_sender, (long) uid);
    if ((status = mail_stream_finish(handle, (VSTRING *) 0)) != 0)
	msg_fatal_status((status & CLEANUP_STAT_BAD) ? EX_SOFTWARE :
			 (status & CLEANUP_STAT_WRITE) ? EX_TEMPFAIL :
			 EX_UNAVAILABLE, "%s(%ld): %s", saved_sender,
			 (long) uid, cleanup_strerror(status));

    /*
     * Don't leave them in the dark.
     */
    if (DEL_REQ_TRACE_FLAGS(flags)) {
	vstream_printf("Mail Delivery Status Report will be mailed to <%s>.\n",
		       saved_sender);
	vstream_fflush(VSTREAM_OUT);
    }

    /*
     * Cleanup. Not really necessary as we're about to exit, but good for
     * debugging purposes.
     */
    vstring_free(buf);
    myfree(saved_sender);
}

/* tempfail - sanitize exit status after library run-time error */

static void tempfail(void)
{
    exit(EX_TEMPFAIL);
}

MAIL_VERSION_STAMP_DECLARE;

/* main - the main program */

int     main(int argc, char **argv)
{
    static char *full_name = 0;		/* sendmail -F */
    struct stat st;
    char   *slash;
    char   *sender = 0;			/* sendmail -f */
    int     c;
    int     fd;
    int     mode;
    ARGV   *ext_argv;
    int     debug_me = 0;
    int     err;
    int     n;
    int     flags = SM_FLAG_DEFAULT;
    char   *site_to_flush = 0;
    char   *id_to_flush = 0;
    char   *encoding = 0;
    char   *qtime = 0;
    const char *errstr;
    uid_t   uid;
    const char *rewrite_context = MAIL_ATTR_RWR_LOCAL;
    int     dsn_notify = 0;
    int     dsn_ret = 0;
    const char *dsn_envid = 0;
    int     saved_optind;
    ARGV   *import_env;

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    /*
     * Be consistent with file permissions.
     */
    umask(022);

    /*
     * To minimize confusion, make sure that the standard file descriptors
     * are open before opening anything else. XXX Work around for 44BSD where
     * fstat can return EBADF on an open file descriptor.
     */
    for (fd = 0; fd < 3; fd++)
	if (fstat(fd, &st) == -1
	    && (close(fd), open("/dev/null", O_RDWR, 0)) != fd)
	    msg_fatal_status(EX_OSERR, "open /dev/null: %m");

    /*
     * The CDE desktop calendar manager leaks a parent file descriptor into
     * the child process. For the sake of sendmail compatibility we have to
     * close the file descriptor otherwise mail notification will hang.
     */
    for ( /* void */ ; fd < 100; fd++)
	(void) close(fd);

    /*
     * Process environment options as early as we can. We might be called
     * from a set-uid (set-gid) program, so be careful with importing
     * environment variables.
     */
    if (safe_getenv(CONF_ENV_VERB))
	msg_verbose = 1;
    if (safe_getenv(CONF_ENV_DEBUG))
	debug_me = 1;

    /*
     * Initialize. Set up logging, read the global configuration file and
     * extract configuration information. Set up signal handlers so that we
     * can clean up incomplete output.
     */
    if ((slash = strrchr(argv[0], '/')) != 0 && slash[1])
	argv[0] = slash + 1;
    msg_vstream_init(argv[0], VSTREAM_ERR);
    msg_cleanup(tempfail);
    msg_syslog_init(mail_task("sendmail"), LOG_PID, LOG_FACILITY);
    set_mail_conf_str(VAR_PROCNAME, var_procname = mystrdup(argv[0]));

    /*
     * Check the Postfix library version as soon as we enable logging.
     */
    MAIL_VERSION_CHECK;

    /*
     * Some sites mistakenly install Postfix sendmail as set-uid root. Drop
     * set-uid privileges only when root, otherwise some systems will not
     * reset the saved set-userid, which would be a security vulnerability.
     */
    if (geteuid() == 0 && getuid() != 0) {
	msg_warn("the Postfix sendmail command has set-uid root file permissions");
	msg_warn("or the command is run from a set-uid root process");
	msg_warn("the Postfix sendmail command must be installed without set-uid root file permissions");
	set_ugid(getuid(), getgid());
    }

    /*
     * Further initialization. Load main.cf first, so that command-line
     * options can override main.cf settings. Pre-scan the argument list so
     * that we load the right main.cf file.
     */
#define GETOPT_LIST "A:B:C:F:GIL:N:O:R:UV:X:b:ce:f:h:imno:p:r:q:tvx"

    saved_optind = optind;
    while (argv[OPTIND] != 0) {
	if (strcmp(argv[OPTIND], "-q") == 0) {	/* not getopt compatible */
	    optind++;
	    continue;
	}
	if ((c = GETOPT(argc, argv, GETOPT_LIST)) <= 0)
	    break;
	if (c == 'C') {
	    VSTRING *buf = vstring_alloc(1);
	    char   *dir;

	    dir = strcmp(sane_basename(buf, optarg), MAIN_CONF_FILE) == 0 ?
		sane_dirname(buf, optarg) : optarg;
	    if (strcmp(dir, DEF_CONFIG_DIR) != 0 && geteuid() != 0)
		mail_conf_checkdir(dir);
	    if (setenv(CONF_ENV_PATH, dir, 1) < 0)
		msg_fatal_status(EX_UNAVAILABLE, "out of memory");
	    vstring_free(buf);
	}
    }
    optind = saved_optind;
    mail_conf_read();
    /* Enforce consistent operation of different Postfix parts.	 */
    import_env = mail_parm_split(VAR_IMPORT_ENVIRON, var_import_environ);
    update_env(import_env->argv);
    argv_free(import_env);
    /* Re-evaluate mail_task() after reading main.cf. */
    msg_syslog_init(mail_task("sendmail"), LOG_PID, LOG_FACILITY);
    get_mail_conf_str_table(str_table);

    if (chdir(var_queue_dir))
	msg_fatal_status(EX_UNAVAILABLE, "chdir %s: %m", var_queue_dir);

    signal(SIGPIPE, SIG_IGN);

    /*
     * Optionally start the debugger on ourself. This must be done after
     * reading the global configuration file, because that file specifies
     * what debugger command to execute.
     */
    if (debug_me)
	debug_process();

    /*
     * The default mode of operation is determined by the process name. It
     * can, however, be changed via command-line options (for example,
     * "newaliases -bp" will show the mail queue).
     */
    if (strcmp(argv[0], "mailq") == 0) {
	mode = SM_MODE_MAILQ;
    } else if (strcmp(argv[0], "newaliases") == 0) {
	mode = SM_MODE_NEWALIAS;
    } else if (strcmp(argv[0], "smtpd") == 0) {
	mode = SM_MODE_DAEMON;
    } else {
	mode = SM_MODE_ENQUEUE;
    }

    /*
     * Parse JCL. Sendmail has been around for a long time, and has acquired
     * a large number of options in the course of time. Some options such as
     * -q are not parsable with GETOPT() and get special treatment.
     */
#define OPTIND  (optind > 0 ? optind : 1)

    while (argv[OPTIND] != 0) {
	if (strcmp(argv[OPTIND], "-q") == 0) {
	    if (mode == SM_MODE_DAEMON)
		msg_warn("ignoring -q option in daemon mode");
	    else
		mode = SM_MODE_FLUSHQ;
	    optind++;
	    continue;
	}
	if (strcmp(argv[OPTIND], "-V") == 0
	    && argv[OPTIND + 1] != 0 && strlen(argv[OPTIND + 1]) == 2) {
	    msg_warn("option -V is deprecated with Postfix 2.3; "
		     "specify -XV instead");
	    argv[OPTIND] = "-XV";
	}
	if (strncmp(argv[OPTIND], "-V", 2) == 0 && strlen(argv[OPTIND]) == 4) {
	    msg_warn("option %s is deprecated with Postfix 2.3; "
		     "specify -X%s instead",
		     argv[OPTIND], argv[OPTIND] + 1);
	    argv[OPTIND] = concatenate("-X", argv[OPTIND] + 1, (char *) 0);
	}
	if (strcmp(argv[OPTIND], "-XV") == 0) {
	    verp_delims = var_verp_delims;
	    optind++;
	    continue;
	}
	if ((c = GETOPT(argc, argv, GETOPT_LIST)) <= 0)
	    break;
	switch (c) {
	default:
	    if (msg_verbose)
		msg_info("-%c option ignored", c);
	    break;
	case 'n':
	    msg_fatal_status(EX_USAGE, "-%c option not supported", c);
	case 'B':
	    if (strcmp(optarg, "8BITMIME") == 0)/* RFC 1652 */
		encoding = MAIL_ATTR_ENC_8BIT;
	    else if (strcmp(optarg, "7BIT") == 0)	/* RFC 1652 */
		encoding = MAIL_ATTR_ENC_7BIT;
	    else
		msg_fatal_status(EX_USAGE, "-B option needs 8BITMIME or 7BIT");
	    break;
	case 'F':				/* full name */
	    full_name = optarg;
	    break;
	case 'G':				/* gateway submission */
	    rewrite_context = MAIL_ATTR_RWR_REMOTE;
	    break;
	case 'I':				/* newaliases */
	    mode = SM_MODE_NEWALIAS;
	    break;
	case 'N':
	    if ((dsn_notify = dsn_notify_mask(optarg)) == 0)
		msg_warn("bad -N option value -- ignored");
	    break;
	case 'R':
	    if ((dsn_ret = dsn_ret_code(optarg)) == 0)
		msg_warn("bad -R option value -- ignored");
	    break;
	case 'V':				/* DSN, was: VERP */
	    if (strlen(optarg) > 100)
		msg_warn("too long -V option value -- ignored");
	    else if (!allprint(optarg))
		msg_warn("bad syntax in -V option value -- ignored");
	    else
		dsn_envid = optarg;
	    break;
	case 'X':
	    switch (*optarg) {
	    default:
		msg_fatal_status(EX_USAGE, "unsupported: -%c%c", c, *optarg);
	    case 'V':				/* VERP */
		if (verp_delims_verify(optarg + 1) != 0)
		    msg_fatal_status(EX_USAGE, "-V requires two characters from %s",
				     var_verp_filter);
		verp_delims = optarg + 1;
		break;
	    }
	    break;
	case 'b':
	    switch (*optarg) {
	    default:
		msg_fatal_status(EX_USAGE, "unsupported: -%c%c", c, *optarg);
	    case 'd':				/* daemon mode */
	    case 'l':				/* daemon mode */
		if (mode == SM_MODE_FLUSHQ)
		    msg_warn("ignoring -q option in daemon mode");
		mode = SM_MODE_DAEMON;
		break;
	    case 'h':				/* print host status */
	    case 'H':				/* flush host status */
		mode = SM_MODE_IGNORE;
		break;
	    case 'i':				/* newaliases */
		mode = SM_MODE_NEWALIAS;
		break;
	    case 'm':				/* deliver mail */
		mode = SM_MODE_ENQUEUE;
		break;
	    case 'p':				/* mailq */
		mode = SM_MODE_MAILQ;
		break;
	    case 's':				/* stand-alone mode */
		mode = SM_MODE_USER;
		break;
	    case 'v':				/* expand recipients */
		flags |= DEL_REQ_FLAG_USR_VRFY;
		break;
	    }
	    break;
	case 'f':
	    sender = optarg;
	    break;
	case 'i':
	    flags &= ~SM_FLAG_AEOF;
	    break;
	case 'o':
	    switch (*optarg) {
	    default:
		if (msg_verbose)
		    msg_info("-%c%c option ignored", c, *optarg);
		break;
	    case 'A':
		if (optarg[1] == 0)
		    msg_fatal_status(EX_USAGE, "-oA requires pathname");
		myfree(var_alias_db_map);
		var_alias_db_map = mystrdup(optarg + 1);
		set_mail_conf_str(VAR_ALIAS_DB_MAP, var_alias_db_map);
		break;
	    case '7':
	    case '8':
		break;
	    case 'i':
		flags &= ~SM_FLAG_AEOF;
		break;
	    case 'm':
		break;
	    }
	    break;
	case 'r':				/* obsoleted by -f */
	    sender = optarg;
	    break;
	case 'q':
	    if (ISDIGIT(optarg[0])) {
		qtime = optarg;
	    } else if (optarg[0] == 'R') {
		site_to_flush = optarg + 1;
		if (*site_to_flush == 0)
		    msg_fatal_status(EX_USAGE, "specify: -qRsitename");
	    } else if (optarg[0] == 'I') {
		id_to_flush = optarg + 1;
		if (*id_to_flush == 0)
		    msg_fatal_status(EX_USAGE, "specify: -qIqueueid");
	    } else {
		msg_fatal_status(EX_USAGE, "-q%c is not implemented",
				 optarg[0]);
	    }
	    break;
	case 't':
	    flags |= SM_FLAG_XRCPT;
	    break;
	case 'v':
	    msg_verbose++;
	    break;
	case '?':
	    msg_fatal_status(EX_USAGE, "usage: %s [options]", argv[0]);
	}
    }

    /*
     * Look for conflicting options and arguments.
     */
    if ((flags & SM_FLAG_XRCPT) && mode != SM_MODE_ENQUEUE)
	msg_fatal_status(EX_USAGE, "-t can be used only in delivery mode");

    if (site_to_flush && mode != SM_MODE_ENQUEUE)
	msg_fatal_status(EX_USAGE, "-qR can be used only in delivery mode");

    if (id_to_flush && mode != SM_MODE_ENQUEUE)
	msg_fatal_status(EX_USAGE, "-qI can be used only in delivery mode");

    if (flags & DEL_REQ_FLAG_USR_VRFY) {
	if (flags & SM_FLAG_XRCPT)
	    msg_fatal_status(EX_USAGE, "-t option cannot be used with -bv");
	if (dsn_notify)
	    msg_fatal_status(EX_USAGE, "-N option cannot be used with -bv");
	if (dsn_ret)
	    msg_fatal_status(EX_USAGE, "-R option cannot be used with -bv");
	if (msg_verbose == 1)
	    msg_fatal_status(EX_USAGE, "-v option cannot be used with -bv");
    }

    /*
     * The -v option plays double duty. One requests verbose delivery, more
     * than one requests verbose logging.
     */
    if (msg_verbose == 1 && mode == SM_MODE_ENQUEUE) {
	msg_verbose = 0;
	flags |= DEL_REQ_FLAG_RECORD;
    }

    /*
     * Start processing. Everything is delegated to external commands.
     */
    if (qtime && mode != SM_MODE_DAEMON)
	exit(0);
    switch (mode) {
    default:
	msg_panic("unknown operation mode: %d", mode);
	/* NOTREACHED */
    case SM_MODE_ENQUEUE:
	if (site_to_flush) {
	    if (argv[OPTIND])
		msg_fatal_status(EX_USAGE, "flush site requires no recipient");
	    ext_argv = argv_alloc(2);
	    argv_add(ext_argv, "postqueue", "-s", site_to_flush, (char *) 0);
	    for (n = 0; n < msg_verbose; n++)
		argv_add(ext_argv, "-v", (char *) 0);
	    argv_terminate(ext_argv);
	    mail_run_replace(var_command_dir, ext_argv->argv);
	    /* NOTREACHED */
	} else if (id_to_flush) {
	    if (argv[OPTIND])
		msg_fatal_status(EX_USAGE, "flush queue_id requires no recipient");
	    ext_argv = argv_alloc(2);
	    argv_add(ext_argv, "postqueue", "-i", id_to_flush, (char *) 0);
	    for (n = 0; n < msg_verbose; n++)
		argv_add(ext_argv, "-v", (char *) 0);
	    argv_terminate(ext_argv);
	    mail_run_replace(var_command_dir, ext_argv->argv);
	    /* NOTREACHED */
	} else {
	    enqueue(flags, encoding, dsn_envid, dsn_ret, dsn_notify,
		    rewrite_context, sender, full_name, argv + OPTIND);
	    exit(0);
	    /* NOTREACHED */
	}
	break;
    case SM_MODE_MAILQ:
	if (argv[OPTIND])
	    msg_fatal_status(EX_USAGE,
			     "display queue mode requires no recipient");
	ext_argv = argv_alloc(2);
	argv_add(ext_argv, "postqueue", "-p", (char *) 0);
	for (n = 0; n < msg_verbose; n++)
	    argv_add(ext_argv, "-v", (char *) 0);
	argv_terminate(ext_argv);
	mail_run_replace(var_command_dir, ext_argv->argv);
	/* NOTREACHED */
    case SM_MODE_FLUSHQ:
	if (argv[OPTIND])
	    msg_fatal_status(EX_USAGE,
			     "flush queue mode requires no recipient");
	ext_argv = argv_alloc(2);
	argv_add(ext_argv, "postqueue", "-f", (char *) 0);
	for (n = 0; n < msg_verbose; n++)
	    argv_add(ext_argv, "-v", (char *) 0);
	argv_terminate(ext_argv);
	mail_run_replace(var_command_dir, ext_argv->argv);
	/* NOTREACHED */
    case SM_MODE_DAEMON:
	if (argv[OPTIND])
	    msg_fatal_status(EX_USAGE, "daemon mode requires no recipient");
	ext_argv = argv_alloc(2);
	argv_add(ext_argv, "postfix", (char *) 0);
	for (n = 0; n < msg_verbose; n++)
	    argv_add(ext_argv, "-v", (char *) 0);
	argv_add(ext_argv, "start", (char *) 0);
	argv_terminate(ext_argv);
	err = (mail_run_background(var_command_dir, ext_argv->argv) < 0);
	argv_free(ext_argv);
	exit(err);
	break;
    case SM_MODE_NEWALIAS:
	if (argv[OPTIND])
	    msg_fatal_status(EX_USAGE,
			 "alias initialization mode requires no recipient");
	if (*var_alias_db_map == 0)
	    return (0);
	ext_argv = argv_alloc(2);
	argv_add(ext_argv, "postalias", (char *) 0);
	for (n = 0; n < msg_verbose; n++)
	    argv_add(ext_argv, "-v", (char *) 0);
	argv_split_append(ext_argv, var_alias_db_map, CHARS_COMMA_SP);
	argv_terminate(ext_argv);
	mail_run_replace(var_command_dir, ext_argv->argv);
	/* NOTREACHED */
    case SM_MODE_USER:
	if (argv[OPTIND])
	    msg_fatal_status(EX_USAGE,
			     "stand-alone mode requires no recipient");
	/* The actual enforcement happens in the postdrop command. */
	if ((errstr = check_user_acl_byuid(VAR_SUBMIT_ACL, var_submit_acl,
					   uid = getuid())) != 0)
	    msg_fatal_status(EX_NOPERM,
			     "User %s(%ld) is not allowed to submit mail",
			     errstr, (long) uid);
	ext_argv = argv_alloc(2);
	argv_add(ext_argv, "smtpd", "-S", (char *) 0);
	for (n = 0; n < msg_verbose; n++)
	    argv_add(ext_argv, "-v", (char *) 0);
	argv_terminate(ext_argv);
	mail_run_replace(var_daemon_dir, ext_argv->argv);
	/* NOTREACHED */
    case SM_MODE_IGNORE:
	exit(0);
	/* NOTREACHED */
    }
}
