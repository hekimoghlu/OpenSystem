/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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
#include <vstream.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <syslog.h>
#ifdef USE_PATHS_H
#include <paths.h>
#endif

/* Utility library. */

#include <msg.h>
#include <msg_vstream.h>
#include <msg_syslog.h>
#include <stringops.h>
#include <clean_env.h>
#include <argv.h>
#include <safe.h>
#include <warn_stat.h>

/* Global library. */

#include <mail_conf.h>
#include <mail_params.h>
#include <mail_version.h>
#include <mail_parm_split.h>

/* Additional installation parameters. */

static char *var_sendmail_path;
static char *var_mailq_path;
static char *var_newalias_path;
static char *var_manpage_dir;
static char *var_sample_dir;
static char *var_readme_dir;
static char *var_html_dir;

/* check_setenv - setenv() with extreme prejudice */

static void check_setenv(char *name, char *value)
{
#define CLOBBER 1
    if (setenv(name, value, CLOBBER) < 0)
	msg_fatal("setenv: %m");
}

MAIL_VERSION_STAMP_DECLARE;

/* main - run administrative script from controlled environment */

int     main(int argc, char **argv)
{
    char   *script;
    struct stat st;
    char   *slash;
    int     fd;
    int     ch;
    ARGV   *import_env;
    static const CONFIG_STR_TABLE str_table[] = {
	VAR_SENDMAIL_PATH, DEF_SENDMAIL_PATH, &var_sendmail_path, 1, 0,
	VAR_MAILQ_PATH, DEF_MAILQ_PATH, &var_mailq_path, 1, 0,
	VAR_NEWALIAS_PATH, DEF_NEWALIAS_PATH, &var_newalias_path, 1, 0,
	VAR_MANPAGE_DIR, DEF_MANPAGE_DIR, &var_manpage_dir, 1, 0,
	VAR_SAMPLE_DIR, DEF_SAMPLE_DIR, &var_sample_dir, 1, 0,
	VAR_README_DIR, DEF_README_DIR, &var_readme_dir, 1, 0,
	VAR_HTML_DIR, DEF_HTML_DIR, &var_html_dir, 1, 0,
	0,
    };
    int     force_single_instance;
    ARGV   *my_argv;

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
     * Set up diagnostics. XXX What if stdin is the system console during
     * boot time? It seems a bad idea to log startup errors to the console.
     * This is UNIX, a system that can run without hand holding.
     */
    if ((slash = strrchr(argv[0], '/')) != 0 && slash[1])
	argv[0] = slash + 1;
    if (isatty(STDERR_FILENO))
	msg_vstream_init(argv[0], VSTREAM_ERR);
    msg_syslog_init(argv[0], LOG_PID, LOG_FACILITY);

    /*
     * Check the Postfix library version as soon as we enable logging.
     */
    MAIL_VERSION_CHECK;

    /*
     * The mail system must be run by the superuser so it can revoke
     * privileges for selected operations. That's right - it takes privileges
     * to toss privileges.
     */
    if (getuid() != 0) {
	msg_error("to submit mail, use the Postfix sendmail command");
	msg_fatal("the postfix command is reserved for the superuser");
    }
    if (unsafe() != 0)
	msg_fatal("the postfix command must not run as a set-uid process");

    /*
     * Parse switches.
     */
    while ((ch = GETOPT(argc, argv, "c:Dv")) > 0) {
	switch (ch) {
	default:
	    msg_fatal("usage: %s [-c config_dir] [-Dv] command", argv[0]);
	case 'c':
	    if (*optarg != '/')
		msg_fatal("-c requires absolute pathname");
	    check_setenv(CONF_ENV_PATH, optarg);
	    break;
	case 'D':
	    check_setenv(CONF_ENV_DEBUG, "");
	    break;
	case 'v':
	    msg_verbose++;
	    check_setenv(CONF_ENV_VERB, "");
	    break;
	}
    }
    force_single_instance = (getenv(CONF_ENV_PATH) != 0);

    /*
     * Copy a bunch of configuration parameters into the environment for easy
     * access by the maintenance shell script.
     */
    mail_conf_read();
    get_mail_conf_str_table(str_table);

    /*
     * Alert the sysadmin that the backwards-compatible settings are still in
     * effect.
     */
    if (var_compat_level < CUR_COMPAT_LEVEL) {
	msg_info("Postfix is running with backwards-compatible default "
		 "settings");
	msg_info("See http://www.postfix.org/COMPATIBILITY_README.html "
		 "for details");
	msg_info("To disable backwards compatibility use \"postconf "
		 VAR_COMPAT_LEVEL "=%d\" and \"postfix reload\"",
		 CUR_COMPAT_LEVEL);
    }

    /*
     * Environment import filter, to enforce consistent behavior whether this
     * command is started by hand, or at system boot time. This is necessary
     * because some shell scripts use environment settings to override
     * main.cf settings.
     */
    import_env = mail_parm_split(VAR_IMPORT_ENVIRON, var_import_environ);
    clean_env(import_env->argv);
    argv_free(import_env);

    check_setenv("PATH", ROOT_PATH);		/* sys_defs.h */
    check_setenv(CONF_ENV_PATH, var_config_dir);/* mail_conf.h */

    check_setenv(VAR_COMMAND_DIR, var_command_dir);	/* main.cf */
    check_setenv(VAR_DAEMON_DIR, var_daemon_dir);	/* main.cf */
    check_setenv(VAR_DATA_DIR, var_data_dir);	/* main.cf */
    check_setenv(VAR_META_DIR, var_meta_dir);	/* main.cf */
    check_setenv(VAR_QUEUE_DIR, var_queue_dir);	/* main.cf */
    check_setenv(VAR_CONFIG_DIR, var_config_dir);	/* main.cf */
    check_setenv(VAR_SHLIB_DIR, var_shlib_dir);	/* main.cf */

    /*
     * Do we want to keep adding things here as shell scripts evolve?
     */
    check_setenv(VAR_MAIL_OWNER, var_mail_owner);	/* main.cf */
    check_setenv(VAR_SGID_GROUP, var_sgid_group);	/* main.cf */
    check_setenv(VAR_SENDMAIL_PATH, var_sendmail_path);	/* main.cf */
    check_setenv(VAR_MAILQ_PATH, var_mailq_path);	/* main.cf */
    check_setenv(VAR_NEWALIAS_PATH, var_newalias_path);	/* main.cf */
    check_setenv(VAR_MANPAGE_DIR, var_manpage_dir);	/* main.cf */
    check_setenv(VAR_SAMPLE_DIR, var_sample_dir);	/* main.cf */
    check_setenv(VAR_README_DIR, var_readme_dir);	/* main.cf */
    check_setenv(VAR_HTML_DIR, var_html_dir);	/* main.cf */

    /*
     * Make sure these directories exist. Run the maintenance scripts with as
     * current directory the mail database.
     */
    if (chdir(var_command_dir))
	msg_fatal("chdir(%s): %m", var_command_dir);
    if (chdir(var_daemon_dir))
	msg_fatal("chdir(%s): %m", var_daemon_dir);
    if (chdir(var_queue_dir))
	msg_fatal("chdir(%s): %m", var_queue_dir);

    /*
     * Run the management script.
     */
    if (force_single_instance
	|| argv_split(var_multi_conf_dirs, CHARS_COMMA_SP)->argc == 0) {
	script = concatenate(var_daemon_dir, "/postfix-script", (char *) 0);
	if (optind < 1)
	    msg_panic("bad optind value");
	argv[optind - 1] = script;
	execvp(script, argv + optind - 1);
	msg_fatal("%s: %m", script);
    }

    /*
     * Hand off control to a multi-instance manager.
     */
    else {
	if (*var_multi_wrapper == 0)
	    msg_fatal("multi-instance support is requested, but %s is empty",
		      VAR_MULTI_WRAPPER);
	my_argv = argv_split(var_multi_wrapper, CHARS_SPACE);
	do {
	    argv_add(my_argv, argv[optind], (char *) 0);
	} while (argv[optind++] != 0);
	execvp(my_argv->argv[0], my_argv->argv);
	msg_fatal("%s: %m", my_argv->argv[0]);
    }
}
