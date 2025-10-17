/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 11, 2021.
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
/* System libraries. */

#include <sys_defs.h>
#include <sys/stat.h>
#include <syslog.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <limits.h>

/* Utility library. */

#include <events.h>
#include <msg.h>
#include <msg_syslog.h>
#include <vstring.h>
#include <mymalloc.h>
#include <iostuff.h>
#include <vstream.h>
#include <stringops.h>
#include <myflock.h>
#include <watchdog.h>
#include <clean_env.h>
#include <argv.h>
#include <safe.h>
#include <set_eugid.h>
#include <set_ugid.h>

/* Global library. */

#include <mail_params.h>
#include <mail_version.h>
#include <debug_process.h>
#include <mail_task.h>
#include <mail_conf.h>
#include <open_lock.h>
#include <inet_proto.h>
#include <mail_parm_split.h>

/* Application-specific. */

#include "master.h"

int     master_detach = 1;

/* master_exit_event - exit for memory leak testing purposes */

static void master_exit_event(int unused_event, void *unused_context)
{
    msg_info("master exit time has arrived");
    exit(0);
}

/* usage - show hint and terminate */

static NORETURN usage(const char *me)
{
    msg_fatal("usage: %s [-c config_dir] [-D (debug)] [-d (don't detach from terminal)] [-e exit_time] [-t (test)] [-v] [-w (wait for initialization)]", me);
}

MAIL_VERSION_STAMP_DECLARE;

/* main - main program */

int     main(int argc, char **argv)
{
    static VSTREAM *lock_fp;
    static VSTREAM *data_lock_fp;
    VSTRING *lock_path;
    VSTRING *data_lock_path;
    off_t   inherited_limit;
    int     debug_me = 0;
    int     ch;
    int     fd;
    int     n;
    int     test_lock = 0;
    VSTRING *why;
    WATCHDOG *watchdog;
    ARGV   *import_env;
    int     wait_flag = 0;
    int     monitor_fd = -1;

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    /*
     * Initialize.
     */
    umask(077);					/* never fails! */

    /*
     * Process environment options as early as we can.
     */
    if (getenv(CONF_ENV_VERB))
	msg_verbose = 1;
    if (getenv(CONF_ENV_DEBUG))
	debug_me = 1;

    /*
     * Don't die when a process goes away unexpectedly.
     */
    signal(SIGPIPE, SIG_IGN);

    /*
     * Strip and save the process name for diagnostics etc.
     */
    var_procname = mystrdup(basename(argv[0]));

    /*
     * When running a child process, don't leak any open files that were
     * leaked to us by our own (privileged) parent process. Descriptors 0-2
     * are taken care of after we have initialized error logging.
     * 
     * Some systems such as AIX have a huge per-process open file limit. In
     * those cases, limit the search for potential file descriptor leaks to
     * just the first couple hundred.
     * 
     * The Debian post-installation script passes an open file descriptor into
     * the master process and waits forever for someone to close it. Because
     * of this we have to close descriptors > 2, and pray that doing so does
     * not break things.
     */
    closefrom(3);

    /*
     * Initialize logging and exit handler.
     */
    msg_syslog_init(mail_task(var_procname), LOG_PID, LOG_FACILITY);

    /*
     * Check the Postfix library version as soon as we enable logging.
     */
    MAIL_VERSION_CHECK;

    /*
     * The mail system must be run by the superuser so it can revoke
     * privileges for selected operations. That's right - it takes privileges
     * to toss privileges.
     */
    if (getuid() != 0)
	msg_fatal("the master command is reserved for the superuser");
    if (unsafe() != 0)
	msg_fatal("the master command must not run as a set-uid process");

    /*
     * Process JCL.
     */
    while ((ch = GETOPT(argc, argv, "c:Dde:tvw")) > 0) {
	switch (ch) {
	case 'c':
	    if (setenv(CONF_ENV_PATH, optarg, 1) < 0)
		msg_fatal("out of memory");
	    break;
	case 'd':
	    master_detach = 0;
	    break;
	case 'e':
	    event_request_timer(master_exit_event, (void *) 0, atoi(optarg));
	    break;
	case 'D':
	    debug_me = 1;
	    break;
	case 't':
	    test_lock = 1;
	    break;
	case 'v':
	    msg_verbose++;
	    break;
	case 'w':
	    wait_flag = 1;
	    break;
	default:
	    usage(argv[0]);
	    /* NOTREACHED */
	}
    }

    /*
     * This program takes no other arguments.
     */
    if (argc > optind)
	usage(argv[0]);

    /*
     * Sanity check.
     */
    if (test_lock && wait_flag)
	msg_fatal("the -t and -w options cannot be used together");

    /*
     * Run a foreground monitor process that returns an exit status of 0 when
     * the child background process reports successful initialization as a
     * daemon process. We use a generous limit in case main/master.cf specify
     * symbolic hosts/ports and the naming service is slow.
     */
#define MASTER_INIT_TIMEOUT	100		/* keep this limit generous */

    if (wait_flag)
	monitor_fd = master_monitor(MASTER_INIT_TIMEOUT);

    /*
     * If started from a terminal, get rid of any tty association. This also
     * means that all errors and warnings must go to the syslog daemon.
     */
    if (master_detach)
	for (fd = 0; fd < 3; fd++) {
	    (void) close(fd);
	    if (open("/dev/null", O_RDWR, 0) != fd)
		msg_fatal("open /dev/null: %m");
	}

    /*
     * Run in a separate process group, so that "postfix stop" can terminate
     * all MTA processes cleanly. Give up if we can't separate from our
     * parent process. We're not supposed to blow away the parent.
     */
    if (debug_me == 0 && master_detach != 0 && setsid() == -1 && getsid(0) != getpid())
	msg_fatal("unable to set session and process group ID: %m");

    /*
     * Make some room for plumbing with file descriptors. XXX This breaks
     * when a service listens on many ports. In order to do this right we
     * must change the master-child interface so that descriptors do not need
     * to have fixed numbers.
     * 
     * In a child we need two descriptors for the flow control pipe, one for
     * child->master status updates and at least one for listening.
     */
    for (n = 0; n < 5; n++) {
	if (close_on_exec(dup(0), CLOSE_ON_EXEC) < 0)
	    msg_fatal("dup(0): %m");
    }

    /*
     * Final initializations. Unfortunately, we must read the global Postfix
     * configuration file after doing command-line processing, so that we get
     * consistent results when we SIGHUP the server to reload configuration
     * files.
     */
    master_vars_init();

    /*
     * In case of multi-protocol support. This needs to be done because
     * master does not invoke mail_params_init() (it was written before that
     * code existed).
     */
    (void) inet_proto_init(VAR_INET_PROTOCOLS, var_inet_protocols);

    /*
     * Environment import filter, to enforce consistent behavior whether
     * Postfix is started by hand, or at system boot time.
     */
    import_env = mail_parm_split(VAR_IMPORT_ENVIRON, var_import_environ);
    clean_env(import_env->argv);
    argv_free(import_env);

    if ((inherited_limit = get_file_limit()) < 0)
	set_file_limit(OFF_T_MAX);

    if (chdir(var_queue_dir))
	msg_fatal("chdir %s: %m", var_queue_dir);

    /*
     * Lock down the master.pid file. In test mode, no file means that it
     * isn't locked.
     */
    lock_path = vstring_alloc(10);
    data_lock_path = vstring_alloc(10);
    why = vstring_alloc(10);

    vstring_sprintf(lock_path, "%s/%s.pid", DEF_PID_DIR, var_procname);
    if (test_lock && access(vstring_str(lock_path), F_OK) < 0)
	exit(0);
    lock_fp = open_lock(vstring_str(lock_path), O_RDWR | O_CREAT, 0644, why);
    if (test_lock)
	exit(lock_fp ? 0 : 1);
    if (lock_fp == 0)
	msg_fatal("open lock file %s: %s",
		  vstring_str(lock_path), vstring_str(why));
    vstream_fprintf(lock_fp, "%*lu\n", (int) sizeof(unsigned long) * 4,
		    (unsigned long) var_pid);
    if (vstream_fflush(lock_fp))
	msg_fatal("cannot update lock file %s: %m", vstring_str(lock_path));
    close_on_exec(vstream_fileno(lock_fp), CLOSE_ON_EXEC);

    /*
     * Lock down the Postfix-writable data directory.
     */
    vstring_sprintf(data_lock_path, "%s/%s.lock", var_data_dir, var_procname);
    set_eugid(var_owner_uid, var_owner_gid);
    data_lock_fp =
	open_lock(vstring_str(data_lock_path), O_RDWR | O_CREAT, 0644, why);
    set_ugid(getuid(), getgid());
    if (data_lock_fp == 0)
	msg_fatal("open lock file %s: %s",
		  vstring_str(data_lock_path), vstring_str(why));
    vstream_fprintf(data_lock_fp, "%*lu\n", (int) sizeof(unsigned long) * 4,
		    (unsigned long) var_pid);
    if (vstream_fflush(data_lock_fp))
	msg_fatal("cannot update lock file %s: %m", vstring_str(data_lock_path));
    close_on_exec(vstream_fileno(data_lock_fp), CLOSE_ON_EXEC);

    /*
     * Clean up.
     */
    vstring_free(why);
    vstring_free(lock_path);
    vstring_free(data_lock_path);

    /*
     * Optionally start the debugger on ourself.
     */
    if (debug_me)
	debug_process();

    /*
     * Finish initialization, last part. We must process configuration files
     * after processing command-line parameters, so that we get consistent
     * results when we SIGHUP the server to reload configuration files.
     */
    master_config();
    master_sigsetup();
    master_flow_init();
    msg_info("daemon started -- version %s, configuration %s",
	     var_mail_version, var_config_dir);

    /*
     * Report successful initialization to the foreground monitor process.
     */
    if (monitor_fd >= 0) {
	write(monitor_fd, "", 1);
	(void) close(monitor_fd);
    }

    /*
     * Process events. The event handler will execute the read/write/timer
     * action routines. Whenever something has happened, see if we received
     * any signal in the mean time. Although the master process appears to do
     * multiple things at the same time, it really is all a single thread, so
     * that there are no concurrency conflicts within the master process.
     */
#define MASTER_WATCHDOG_TIME	1000

    watchdog = watchdog_create(MASTER_WATCHDOG_TIME, (WATCHDOG_FN) 0, (void *) 0);
    for (;;) {
#ifdef HAS_VOLATILE_LOCKS
	if (myflock(vstream_fileno(lock_fp), INTERNAL_LOCK,
		    MYFLOCK_OP_EXCLUSIVE) < 0)
	    msg_fatal("refresh exclusive lock: %m");
	if (myflock(vstream_fileno(data_lock_fp), INTERNAL_LOCK,
		    MYFLOCK_OP_EXCLUSIVE) < 0)
	    msg_fatal("refresh exclusive lock: %m");
#endif
	watchdog_start(watchdog);		/* same as trigger servers */
	event_loop(MASTER_WATCHDOG_TIME / 2);
	if (master_gotsighup) {
	    msg_info("reload -- version %s, configuration %s",
		     var_mail_version, var_config_dir);
	    master_gotsighup = 0;		/* this first */
	    master_vars_init();			/* then this */
	    master_refresh();			/* then this */
	}
	if (master_gotsigchld) {
	    if (msg_verbose)
		msg_info("got sigchld");
	    master_gotsigchld = 0;		/* this first */
	    master_reap_child();		/* then this */
	}
    }
}
