/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 10, 2022.
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
#include <stdlib.h>
#include <stdio.h>			/* remove() */
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <syslog.h>
#include <errno.h>
#include <warn_stat.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <vstream.h>
#include <vstring.h>
#include <msg_vstream.h>
#include <msg_syslog.h>
#include <argv.h>
#include <iostuff.h>
#include <stringops.h>

/* Global library. */

#include <mail_proto.h>
#include <mail_queue.h>
#include <mail_params.h>
#include <mail_version.h>
#include <mail_conf.h>
#include <mail_task.h>
#include <clean_env.h>
#include <mail_stream.h>
#include <cleanup_user.h>
#include <record.h>
#include <rec_type.h>
#include <mail_dict.h>
#include <user_acl.h>
#include <rec_attr_map.h>
#include <mail_parm_split.h>

/* Application-specific. */

 /*
  * WARNING WARNING WARNING
  * 
  * This software is designed to run set-gid. In order to avoid exploitation of
  * privilege, this software should not run any external commands, nor should
  * it take any information from the user unless that information can be
  * properly sanitized. To get an idea of how much information a process can
  * inherit from a potentially hostile user, examine all the members of the
  * process structure (typically, in /usr/include/sys/proc.h): the current
  * directory, open files, timers, signals, environment, command line, umask,
  * and so on.
  */

 /*
  * Local mail submission access list.
  */
char   *var_submit_acl;

static const CONFIG_STR_TABLE str_table[] = {
    VAR_SUBMIT_ACL, DEF_SUBMIT_ACL, &var_submit_acl, 0, 0,
    0,
};

 /*
  * Queue file name. Global, so that the cleanup routine can find it when
  * called by the run-time error handler.
  */
static char *postdrop_path;

/* postdrop_sig - catch signal and clean up */

static void postdrop_sig(int sig)
{

    /*
     * This is the fatal error handler. Don't try to do anything fancy.
     * 
     * msg_vstream does not allocate memory, but msg_syslog may indirectly in
     * syslog(), so it should not be called from a user-triggered signal
     * handler.
     * 
     * Assume atomic signal() updates, even when emulated with sigaction(). We
     * use the in-kernel SIGINT handler address as an atomic variable to
     * prevent nested postdrop_sig() calls. For this reason, main() must
     * configure postdrop_sig() as SIGINT handler before other signal
     * handlers are allowed to invoke postdrop_sig().
     */
    if (signal(SIGINT, SIG_IGN) != SIG_IGN) {
	(void) signal(SIGQUIT, SIG_IGN);
	(void) signal(SIGTERM, SIG_IGN);
	(void) signal(SIGHUP, SIG_IGN);
	if (postdrop_path) {
	    (void) remove(postdrop_path);
	    postdrop_path = 0;
	}
	/* Future proofing. If you need exit() here then you broke Postfix. */
	if (sig)
	    _exit(sig);
    }
}

/* postdrop_cleanup - callback for the runtime error handler */

static void postdrop_cleanup(void)
{
    postdrop_sig(0);
}

MAIL_VERSION_STAMP_DECLARE;

/* main - the main program */

int     main(int argc, char **argv)
{
    struct stat st;
    int     fd;
    int     c;
    VSTRING *buf;
    int     status;
    MAIL_STREAM *dst;
    int     rec_type;
    static char *segment_info[] = {
	REC_TYPE_POST_ENVELOPE, REC_TYPE_POST_CONTENT, REC_TYPE_POST_EXTRACT, ""
    };
    char  **expected;
    uid_t   uid = getuid();
    ARGV   *import_env;
    const char *error_text;
    char   *attr_name;
    char   *attr_value;
    const char *errstr;
    char   *junk;
    struct timeval start;
    int     saved_errno;
    int     from_count = 0;
    int     rcpt_count = 0;
    int     validate_input = 1;

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
	    msg_fatal("open /dev/null: %m");

    /*
     * Set up logging. Censor the process name: it is provided by the user.
     */
    argv[0] = "postdrop";
    msg_vstream_init(argv[0], VSTREAM_ERR);
    msg_syslog_init(mail_task("postdrop"), LOG_PID, LOG_FACILITY);
    set_mail_conf_str(VAR_PROCNAME, var_procname = mystrdup(argv[0]));

    /*
     * Check the Postfix library version as soon as we enable logging.
     */
    MAIL_VERSION_CHECK;

    /*
     * Parse JCL. This program is set-gid and must sanitize all command-line
     * arguments. The configuration directory argument is validated by the
     * mail configuration read routine. Don't do complex things until we have
     * completed initializations.
     */
    while ((c = GETOPT(argc, argv, "c:rv")) > 0) {
	switch (c) {
	case 'c':
	    if (setenv(CONF_ENV_PATH, optarg, 1) < 0)
		msg_fatal("out of memory");
	    break;
	case 'r':				/* forward compatibility */
	    break;
	case 'v':
	    if (geteuid() == 0)
		msg_verbose++;
	    break;
	default:
	    msg_fatal("usage: %s [-c config_dir] [-v]", argv[0]);
	}
    }

    /*
     * Read the global configuration file and extract configuration
     * information. Some claim that the user should supply the working
     * directory instead. That might be OK, given that this command needs
     * write permission in a subdirectory called "maildrop". However we still
     * need to reliably detect incomplete input, and so we must perform
     * record-level I/O. With that, we should also take the opportunity to
     * perform some sanity checks on the input.
     */
    mail_conf_read();
    /* Re-evaluate mail_task() after reading main.cf. */
    msg_syslog_init(mail_task("postdrop"), LOG_PID, LOG_FACILITY);
    get_mail_conf_str_table(str_table);

    /*
     * Mail submission access control. Should this be in the user-land gate,
     * or in the daemon process?
     */
    mail_dict_init();
    if ((errstr = check_user_acl_byuid(VAR_SUBMIT_ACL, var_submit_acl,
				       uid)) != 0)
	msg_fatal("User %s(%ld) is not allowed to submit mail",
		  errstr, (long) uid);

    /*
     * Stop run-away process accidents by limiting the queue file size. This
     * is not a defense against DOS attack.
     */
    if (var_message_limit > 0 && get_file_limit() > var_message_limit)
	set_file_limit((off_t) var_message_limit);

    /*
     * This program is installed with setgid privileges. Strip the process
     * environment so that we don't have to trust the C library.
     */
    import_env = mail_parm_split(VAR_IMPORT_ENVIRON, var_import_environ);
    clean_env(import_env->argv);
    argv_free(import_env);

    if (chdir(var_queue_dir))
	msg_fatal("chdir %s: %m", var_queue_dir);
    if (msg_verbose)
	msg_info("chdir %s", var_queue_dir);

    /*
     * Set up signal handlers and a runtime error handler so that we can
     * clean up incomplete output.
     * 
     * postdrop_sig() uses the in-kernel SIGINT handler address as an atomic
     * variable to prevent nested postdrop_sig() calls. For this reason, the
     * SIGINT handler must be configured before other signal handlers are
     * allowed to invoke postdrop_sig().
     */
    signal(SIGPIPE, SIG_IGN);
    signal(SIGXFSZ, SIG_IGN);

    signal(SIGINT, postdrop_sig);
    signal(SIGQUIT, postdrop_sig);
    if (signal(SIGTERM, SIG_IGN) == SIG_DFL)
	signal(SIGTERM, postdrop_sig);
    if (signal(SIGHUP, SIG_IGN) == SIG_DFL)
	signal(SIGHUP, postdrop_sig);
    msg_cleanup(postdrop_cleanup);

    /* End of initializations. */

    /*
     * Don't trust the caller's time information.
     */
    GETTIMEOFDAY(&start);

    /*
     * Create queue file. mail_stream_file() never fails. Send the queue ID
     * to the caller. Stash away a copy of the queue file name so we can
     * clean up in case of a fatal error or an interrupt.
     */
    dst = mail_stream_file(MAIL_QUEUE_MAILDROP, MAIL_CLASS_PUBLIC,
			   var_pickup_service, 0444);
    attr_print(VSTREAM_OUT, ATTR_FLAG_NONE,
	       SEND_ATTR_STR(MAIL_ATTR_QUEUEID, dst->id),
	       ATTR_TYPE_END);
    vstream_fflush(VSTREAM_OUT);
    postdrop_path = mystrdup(VSTREAM_PATH(dst->stream));

    /*
     * Copy stdin to file. The format is checked so that we can recognize
     * incomplete input and cancel the operation. With the sanity checks
     * applied here, the pickup daemon could skip format checks and pass a
     * file descriptor to the cleanup daemon. These are by no means all
     * sanity checks - the cleanup service and queue manager services will
     * reject messages that lack required information.
     * 
     * If something goes wrong, slurp up the input before responding to the
     * client, otherwise the client will give up after detecting SIGPIPE.
     * 
     * Allow attribute records if the attribute specifies the MIME body type
     * (sendmail -B).
     */
    vstream_control(VSTREAM_IN, CA_VSTREAM_CTL_PATH("stdin"), CA_VSTREAM_CTL_END);
    buf = vstring_alloc(100);
    expected = segment_info;
    /* Override time information from the untrusted caller. */
    rec_fprintf(dst->stream, REC_TYPE_TIME, REC_TYPE_TIME_FORMAT,
		REC_TYPE_TIME_ARG(start));
    for (;;) {
	/* Don't allow PTR records. */
	rec_type = rec_get_raw(VSTREAM_IN, buf, var_line_limit, REC_FLAG_NONE);
	if (rec_type == REC_TYPE_EOF) {		/* request cancelled */
	    mail_stream_cleanup(dst);
	    if (remove(postdrop_path))
		msg_warn("uid=%ld: remove %s: %m", (long) uid, postdrop_path);
	    else if (msg_verbose)
		msg_info("remove %s", postdrop_path);
	    myfree(postdrop_path);
	    postdrop_path = 0;
	    exit(0);
	}
	if (rec_type == REC_TYPE_ERROR)
	    msg_fatal("uid=%ld: malformed input", (long) uid);
	if (strchr(*expected, rec_type) == 0)
	    msg_fatal("uid=%ld: unexpected record type: %d", (long) uid, rec_type);
	if (rec_type == **expected)
	    expected++;
	/* Override time information from the untrusted caller. */
	if (rec_type == REC_TYPE_TIME)
	    continue;
	/* Check these at submission time instead of pickup time. */
	if (rec_type == REC_TYPE_FROM)
	    from_count++;
	if (rec_type == REC_TYPE_RCPT)
	    rcpt_count++;
	/* Limit the attribute types that users may specify. */
	if (rec_type == REC_TYPE_ATTR) {
	    if ((error_text = split_nameval(vstring_str(buf), &attr_name,
					    &attr_value)) != 0) {
		msg_warn("uid=%ld: ignoring malformed record: %s: %.200s",
			 (long) uid, error_text, vstring_str(buf));
		continue;
	    }
#define STREQ(x,y) (strcmp(x,y) == 0)

	    if ((STREQ(attr_name, MAIL_ATTR_ENCODING)
		 && (STREQ(attr_value, MAIL_ATTR_ENC_7BIT)
		     || STREQ(attr_value, MAIL_ATTR_ENC_8BIT)
		     || STREQ(attr_value, MAIL_ATTR_ENC_NONE)))
		|| STREQ(attr_name, MAIL_ATTR_DSN_ENVID)
		|| STREQ(attr_name, MAIL_ATTR_DSN_NOTIFY)
		|| rec_attr_map(attr_name)
		|| (STREQ(attr_name, MAIL_ATTR_RWR_CONTEXT)
		    && (STREQ(attr_value, MAIL_ATTR_RWR_LOCAL)
			|| STREQ(attr_value, MAIL_ATTR_RWR_REMOTE)))
		|| STREQ(attr_name, MAIL_ATTR_TRACE_FLAGS)) {	/* XXX */
		rec_fprintf(dst->stream, REC_TYPE_ATTR, "%s=%s",
			    attr_name, attr_value);
	    } else {
		msg_warn("uid=%ld: ignoring attribute record: %.200s=%.200s",
			 (long) uid, attr_name, attr_value);
	    }
	    continue;
	}
	if (REC_PUT_BUF(dst->stream, rec_type, buf) < 0) {
	    /* rec_get() errors must not clobber errno. */
	    saved_errno = errno;
	    while ((rec_type = rec_get_raw(VSTREAM_IN, buf, var_line_limit,
					   REC_FLAG_NONE)) != REC_TYPE_END
		   && rec_type != REC_TYPE_EOF)
		if (rec_type == REC_TYPE_ERROR)
		    msg_fatal("uid=%ld: malformed input", (long) uid);
	    validate_input = 0;
	    errno = saved_errno;
	    break;
	}
	if (rec_type == REC_TYPE_END)
	    break;
    }
    vstring_free(buf);

    /*
     * As of Postfix 2.7 the pickup daemon discards mail without recipients.
     * Such mail may enter the maildrop queue when "postsuper -r" is invoked
     * before the queue manager deletes an already delivered message. Looking
     * at file ownership is not a good way to make decisions on what mail to
     * discard. Instead, the pickup server now requires that new submissions
     * always have at least one recipient record.
     * 
     * The Postfix sendmail command already rejects mail without recipients.
     * However, in the future postdrop may receive mail via other programs,
     * so we add a redundant recipient check here for future proofing.
     * 
     * The test for the sender address is just for consistency of error
     * reporting (report at submission time instead of pickup time). Besides
     * the segment terminator records, there aren't any other mandatory
     * records in a Postfix submission queue file.
     */
    if (validate_input && (from_count == 0 || rcpt_count == 0)) {
	status = CLEANUP_STAT_BAD;
	mail_stream_cleanup(dst);
    }

    /*
     * Finish the file.
     */
    else if ((status = mail_stream_finish(dst, (VSTRING *) 0)) != 0) {
	msg_warn("uid=%ld: %m", (long) uid);
	postdrop_cleanup();
    }

    /*
     * Disable deletion on fatal error before reporting success, so the file
     * will not be deleted after we have taken responsibility for delivery.
     */
    if (postdrop_path) {
	junk = postdrop_path;
	postdrop_path = 0;
	myfree(junk);
    }

    /*
     * Send the completion status to the caller and terminate.
     */
    attr_print(VSTREAM_OUT, ATTR_FLAG_NONE,
	       SEND_ATTR_INT(MAIL_ATTR_STATUS, status),
	       SEND_ATTR_STR(MAIL_ATTR_WHY, ""),
	       ATTR_TYPE_END);
    vstream_fflush(VSTREAM_OUT);
    exit(status);
}
