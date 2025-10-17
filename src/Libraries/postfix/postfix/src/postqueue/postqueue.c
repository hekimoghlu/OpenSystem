/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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
#include <stdlib.h>
#include <signal.h>
#include <sysexits.h>
#include <errno.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <clean_env.h>
#include <vstream.h>
#include <msg_vstream.h>
#include <msg_syslog.h>
#include <argv.h>
#include <safe.h>
#include <connect.h>
#include <valid_hostname.h>
#include <warn_stat.h>
#include <events.h>
#include <stringops.h>

/* Global library. */

#include <mail_proto.h>
#include <mail_params.h>
#include <mail_version.h>
#include <mail_conf.h>
#include <mail_task.h>
#include <mail_run.h>
#include <mail_flush.h>
#include <mail_queue.h>
#include <flush_clnt.h>
#include <smtp_stream.h>
#include <user_acl.h>
#include <valid_mailhost_addr.h>
#include <mail_dict.h>
#include <mail_parm_split.h>

/* Application-specific. */

#include <postqueue.h>

 /*
  * WARNING WARNING WARNING
  * 
  * This software is designed to run set-gid. In order to avoid exploitation of
  * privilege, this software should not run any external commands, nor should
  * it take any information from the user, unless that information can be
  * properly sanitized. To get an idea of how much information a process can
  * inherit from a potentially hostile user, examine all the members of the
  * process structure (typically, in /usr/include/sys/proc.h): the current
  * directory, open files, timers, signals, environment, command line, umask,
  * and so on.
  */

 /*
  * Modes of operation.
  * 
  * XXX To support flush by recipient domain, or for destinations that have no
  * mapping to logfile, the server has to defend against resource exhaustion
  * attacks. A malicious user could fork off a postqueue client that starts
  * an expensive requests and then kills the client immediately; this way she
  * could create a high Postfix load on the system without ever exceeding her
  * own per-user process limit. To prevent this, either the server needs to
  * establish frequent proof of client liveliness with challenge/response, or
  * the client needs to restrict expensive requests to privileged users only.
  * 
  * We don't have this problem with queue listings. The showq server detects an
  * EPIPE error after reporting a few queue entries.
  */
#define PQ_MODE_DEFAULT		0	/* noop */
#define PQ_MODE_MAILQ_LIST	1	/* list mail queue */
#define PQ_MODE_FLUSH_QUEUE	2	/* flush queue */
#define PQ_MODE_FLUSH_SITE	3	/* flush site */
#define PQ_MODE_FLUSH_FILE	4	/* flush message */
#define PQ_MODE_JSON_LIST	5	/* JSON-format queue listing */

 /*
  * Silly little macros (SLMs).
  */
#define STR	vstring_str

 /*
  * Queue manipulation access lists.
  */
char   *var_flush_acl;
char   *var_showq_acl;

static const CONFIG_STR_TABLE str_table[] = {
    VAR_FLUSH_ACL, DEF_FLUSH_ACL, &var_flush_acl, 0, 0,
    VAR_SHOWQ_ACL, DEF_SHOWQ_ACL, &var_showq_acl, 0, 0,
    0,
};

/* show_queue - show queue status */

static void show_queue(int mode)
{
    const char *errstr;
    VSTREAM *showq;
    int     n;
    uid_t   uid = getuid();

    if (uid != 0 && uid != var_owner_uid
	&& (errstr = check_user_acl_byuid(VAR_SHOWQ_ACL, var_showq_acl,
					  uid)) != 0)
	msg_fatal_status(EX_NOPERM,
		       "User %s(%ld) is not allowed to view the mail queue",
			 errstr, (long) uid);

    /*
     * Connect to the show queue service.
     */
    if ((showq = mail_connect(MAIL_CLASS_PUBLIC, var_showq_service, BLOCKING)) != 0) {
	switch (mode) {
	case PQ_MODE_MAILQ_LIST:
	    showq_compat(showq);
	    break;
	case PQ_MODE_JSON_LIST:
	    showq_json(showq);
	    break;
	default:
	    msg_panic("show_queue: unknown mode %d", mode);
	}
	if (vstream_fclose(showq))
	    msg_warn("close: %m");
    }

    /*
     * Don't assume that the mail system is down when the user has
     * insufficient permission to access the showq socket.
     */
    else if (errno == EACCES) {
	msg_fatal_status(EX_SOFTWARE,
			 "Connect to the %s %s service: %m",
			 var_mail_name, var_showq_service);
    }

    /*
     * When the mail system is down, the superuser can still access the queue
     * directly. Just run the showq program in stand-alone mode.
     */
    else if (geteuid() == 0) {
	char   *showq_path;
	ARGV   *argv;
	int     stat;

	msg_warn("Mail system is down -- accessing queue directly");
	showq_path = concatenate(var_daemon_dir, "/", var_showq_service,
				 (char *) 0);
	argv = argv_alloc(6);
	argv_add(argv, showq_path, "-u", "-S", (char *) 0);
	for (n = 0; n < msg_verbose; n++)
	    argv_add(argv, "-v", (char *) 0);
	argv_terminate(argv);
	if ((showq = vstream_popen(O_RDONLY,
				   CA_VSTREAM_POPEN_ARGV(argv->argv),
				   CA_VSTREAM_POPEN_END)) == 0) {
	    stat = -1;
	} else {
	    switch (mode) {
	    case PQ_MODE_MAILQ_LIST:
		showq_compat(showq);
		break;
	    case PQ_MODE_JSON_LIST:
		showq_json(showq);
		break;
	    default:
		msg_panic("show_queue: unknown mode %d", mode);
	    }
	    stat = vstream_pclose(showq);
	}
	argv_free(argv);
	if (stat != 0)
	    msg_fatal_status(stat < 0 ? EX_OSERR : EX_SOFTWARE,
			     "Error running %s", showq_path);
	myfree(showq_path);
    }

    /*
     * When the mail system is down, unprivileged users are stuck, because by
     * design the mail system contains no set_uid programs. The only way for
     * an unprivileged user to cross protection boundaries is to talk to the
     * showq daemon.
     */
    else {
	msg_fatal_status(EX_UNAVAILABLE,
			 "Queue report unavailable - mail system is down");
    }
}

/* flush_queue - force delivery */

static void flush_queue(void)
{
    const char *errstr;
    uid_t   uid = getuid();

    if (uid != 0 && uid != var_owner_uid
	&& (errstr = check_user_acl_byuid(VAR_FLUSH_ACL, var_flush_acl,
					  uid)) != 0)
	msg_fatal_status(EX_NOPERM,
		      "User %s(%ld) is not allowed to flush the mail queue",
			 errstr, (long) uid);

    /*
     * Trigger the flush queue service.
     */
    if (mail_flush_deferred() < 0)
	msg_fatal_status(EX_UNAVAILABLE,
			 "Cannot flush mail queue - mail system is down");
    if (mail_flush_maildrop() < 0)
	msg_fatal_status(EX_UNAVAILABLE,
			 "Cannot flush mail queue - mail system is down");
    event_drain(2);
}

/* flush_site - flush mail for site */

static void flush_site(const char *site)
{
    int     status;
    const char *errstr;
    uid_t   uid = getuid();

    if (uid != 0 && uid != var_owner_uid
	&& (errstr = check_user_acl_byuid(VAR_FLUSH_ACL, var_flush_acl,
					  uid)) != 0)
	msg_fatal_status(EX_NOPERM,
		      "User %s(%ld) is not allowed to flush the mail queue",
			 errstr, (long) uid);

    flush_init();

    switch (status = flush_send_site(site)) {
    case FLUSH_STAT_OK:
	exit(0);
    case FLUSH_STAT_BAD:
	msg_fatal_status(EX_USAGE, "Invalid request: \"%s\"", site);
    case FLUSH_STAT_FAIL:
	msg_fatal_status(EX_UNAVAILABLE,
			 "Cannot flush mail queue - mail system is down");
    case FLUSH_STAT_DENY:
	msg_fatal_status(EX_UNAVAILABLE,
		   "Flush service is not configured for destination \"%s\"",
			 site);
    default:
	msg_fatal_status(EX_SOFTWARE,
			 "Unknown flush server reply status %d", status);
    }
}

/* flush_file - flush mail with specific queue ID */

static void flush_file(const char *queue_id)
{
    int     status;
    const char *errstr;
    uid_t   uid = getuid();

    if (uid != 0 && uid != var_owner_uid
	&& (errstr = check_user_acl_byuid(VAR_FLUSH_ACL, var_flush_acl,
					  uid)) != 0)
	msg_fatal_status(EX_NOPERM,
		      "User %s(%ld) is not allowed to flush the mail queue",
			 errstr, (long) uid);

    switch (status = flush_send_file(queue_id)) {
    case FLUSH_STAT_OK:
	exit(0);
    case FLUSH_STAT_BAD:
	msg_fatal_status(EX_USAGE, "Invalid request: \"%s\"", queue_id);
    case FLUSH_STAT_FAIL:
	msg_fatal_status(EX_UNAVAILABLE,
			 "Cannot flush mail queue - mail system is down");
    default:
	msg_fatal_status(EX_SOFTWARE,
			 "Unexpected flush server reply status %d", status);
    }
}

/* unavailable - sanitize exit status from library run-time errors */

static void unavailable(void)
{
    exit(EX_UNAVAILABLE);
}

/* usage - scream and die */

static NORETURN usage(void)
{
    msg_fatal_status(EX_USAGE, "usage: postqueue -f | postqueue -i queueid | postqueue -j | postqueue -p | postqueue -s site");
}

MAIL_VERSION_STAMP_DECLARE;

/* main - the main program */

int     main(int argc, char **argv)
{
    struct stat st;
    char   *slash;
    int     c;
    int     fd;
    int     mode = PQ_MODE_DEFAULT;
    char   *site_to_flush = 0;
    char   *id_to_flush = 0;
    ARGV   *import_env;
    int     bad_site;

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
	    msg_fatal_status(EX_UNAVAILABLE, "open /dev/null: %m");

    /*
     * Initialize. Set up logging, read the global configuration file and
     * extract configuration information. Set up signal handlers so that we
     * can clean up incomplete output.
     */
    if ((slash = strrchr(argv[0], '/')) != 0 && slash[1])
	argv[0] = slash + 1;
    msg_vstream_init(argv[0], VSTREAM_ERR);
    msg_cleanup(unavailable);
    msg_syslog_init(mail_task("postqueue"), LOG_PID, LOG_FACILITY);
    set_mail_conf_str(VAR_PROCNAME, var_procname = mystrdup(argv[0]));

    /*
     * Check the Postfix library version as soon as we enable logging.
     */
    MAIL_VERSION_CHECK;

    /*
     * Parse JCL. This program is set-gid and must sanitize all command-line
     * parameters. The configuration directory argument is validated by the
     * mail configuration read routine. Don't do complex things until we have
     * completed initializations.
     */
    while ((c = GETOPT(argc, argv, "c:fi:jps:v")) > 0) {
	switch (c) {
	case 'c':				/* non-default configuration */
	    if (setenv(CONF_ENV_PATH, optarg, 1) < 0)
		msg_fatal_status(EX_UNAVAILABLE, "out of memory");
	    break;
	case 'f':				/* flush queue */
	    if (mode != PQ_MODE_DEFAULT)
		usage();
	    mode = PQ_MODE_FLUSH_QUEUE;
	    break;
	case 'i':				/* flush queue file */
	    if (mode != PQ_MODE_DEFAULT)
		usage();
	    mode = PQ_MODE_FLUSH_FILE;
	    id_to_flush = optarg;
	    break;
	case 'j':
	    if (mode != PQ_MODE_DEFAULT)
		usage();
	    mode = PQ_MODE_JSON_LIST;
	    break;
	case 'p':				/* traditional mailq */
	    if (mode != PQ_MODE_DEFAULT)
		usage();
	    mode = PQ_MODE_MAILQ_LIST;
	    break;
	case 's':				/* flush site */
	    if (mode != PQ_MODE_DEFAULT)
		usage();
	    mode = PQ_MODE_FLUSH_SITE;
	    site_to_flush = optarg;
	    break;
	case 'v':
	    if (geteuid() == 0)
		msg_verbose++;
	    break;
	default:
	    usage();
	}
    }
    if (argc > optind)
	usage();

    /*
     * Further initialization...
     */
    mail_conf_read();
    /* Re-evaluate mail_task() after reading main.cf. */
    msg_syslog_init(mail_task("postqueue"), LOG_PID, LOG_FACILITY);
    mail_dict_init();				/* proxy, sql, ldap */
    get_mail_conf_str_table(str_table);

    /*
     * This program is designed to be set-gid, which makes it a potential
     * target for attack. Strip and optionally override the process
     * environment so that we don't have to trust the C library.
     */
    import_env = mail_parm_split(VAR_IMPORT_ENVIRON, var_import_environ);
    clean_env(import_env->argv);
    argv_free(import_env);

    if (chdir(var_queue_dir))
	msg_fatal_status(EX_UNAVAILABLE, "chdir %s: %m", var_queue_dir);

    signal(SIGPIPE, SIG_IGN);

    /* End of initializations. */

    /*
     * Further input validation.
     */
    if (site_to_flush != 0) {
	bad_site = 0;
	if (*site_to_flush == '[') {
	    bad_site = !valid_mailhost_literal(site_to_flush, DONT_GRIPE);
	} else {
	    bad_site = !valid_hostname(site_to_flush, DONT_GRIPE);
	}
	if (bad_site)
	    msg_fatal_status(EX_USAGE,
	      "Cannot flush mail queue - invalid destination: \"%.100s%s\"",
		   site_to_flush, strlen(site_to_flush) > 100 ? "..." : "");
    }
    if (id_to_flush != 0) {
	if (!mail_queue_id_ok(id_to_flush))
	    msg_fatal_status(EX_USAGE,
		       "Cannot flush queue ID - invalid name: \"%.100s%s\"",
		       id_to_flush, strlen(id_to_flush) > 100 ? "..." : "");
    }

    /*
     * Start processing.
     */
    switch (mode) {
    default:
	msg_panic("unknown operation mode: %d", mode);
	/* NOTREACHED */
    case PQ_MODE_MAILQ_LIST:
    case PQ_MODE_JSON_LIST:
	show_queue(mode);
	exit(0);
	break;
    case PQ_MODE_FLUSH_SITE:
	flush_site(site_to_flush);
	exit(0);
	break;
    case PQ_MODE_FLUSH_FILE:
	flush_file(id_to_flush);
	exit(0);
	break;
    case PQ_MODE_FLUSH_QUEUE:
	flush_queue();
	exit(0);
	break;
    case PQ_MODE_DEFAULT:
	usage();
	/* NOTREACHED */
    }
}
