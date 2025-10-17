/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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
#include <string.h>
#include <syslog.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef STRCASECMP_IN_STRINGS_H
#include <strings.h>
#endif

/* Utility library. */

#include <msg.h>
#include <vstring.h>
#include <vstream.h>
#include <vstring_vstream.h>
#include <msg_output.h>
#include <msg_vstream.h>
#include <msg_syslog.h>
#include <warn_stat.h>
#include <clean_env.h>

/* Global library. */

#include <mail_params.h>		/* XXX right place for LOG_FACILITY? */
#include <mail_version.h>
#include <mail_conf.h>
#include <mail_task.h>
#include <mail_parm_split.h>

/* Application-specific. */

 /*
  * Support for the severity level mapping.
  */
struct level_table {
    char   *name;
    int     level;
};

static struct level_table level_table[] = {
    "info", MSG_INFO,
    "warn", MSG_WARN,
    "warning", MSG_WARN,
    "error", MSG_ERROR,
    "err", MSG_ERROR,
    "fatal", MSG_FATAL,
    "crit", MSG_FATAL,
    "panic", MSG_PANIC,
    0,
};

/* level_map - lookup facility or severity value */

static int level_map(char *name)
{
    struct level_table *t;

    for (t = level_table; t->name; t++)
	if (strcasecmp(t->name, name) == 0)
	    return (t->level);
    msg_fatal("bad severity: \"%s\"", name);
}

/* log_argv - log the command line */

static void log_argv(int level, char **argv)
{
    VSTRING *buf = vstring_alloc(100);

    while (*argv) {
	vstring_strcat(buf, *argv++);
	if (*argv)
	    vstring_strcat(buf, " ");
    }
    msg_text(level, vstring_str(buf));
    vstring_free(buf);
}

/* log_stream - log lines from a stream */

static void log_stream(int level, VSTREAM *fp)
{
    VSTRING *buf = vstring_alloc(100);

    while (vstring_get_nonl(buf, fp) != VSTREAM_EOF)
	msg_text(level, vstring_str(buf));
    vstring_free(buf);
}

MAIL_VERSION_STAMP_DECLARE;

/* main - logger */

int     main(int argc, char **argv)
{
    struct stat st;
    int     fd;
    int     ch;
    const char *tag;
    int     log_flags = 0;
    int     level = MSG_INFO;
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
	    msg_fatal("open /dev/null: %m");

    /*
     * Set up diagnostics.
     */
    tag = mail_task(argv[0]);
    if (isatty(STDERR_FILENO))
	msg_vstream_init(tag, VSTREAM_ERR);
    msg_syslog_init(tag, LOG_PID, LOG_FACILITY);

    /*
     * Check the Postfix library version as soon as we enable logging.
     */
    MAIL_VERSION_CHECK;

    /*
     * Parse switches.
     */
    tag = 0;
    while ((ch = GETOPT(argc, argv, "c:ip:t:v")) > 0) {
	switch (ch) {
	default:
	    msg_fatal("usage: %s [-c config_dir] [-i] [-p priority] [-t tag] [-v] [text]", argv[0]);
	    break;
	case 'c':
	    if (setenv(CONF_ENV_PATH, optarg, 1) < 0)
		msg_fatal("out of memory");
	    break;
	case 'i':
	    log_flags |= LOG_PID;
	    break;
	case 'p':
	    level = level_map(optarg);
	    break;
	case 't':
	    tag = optarg;
	    break;
	case 'v':
	    msg_verbose++;
	    break;
	}
    }

    /*
     * Process the main.cf file. This may change the syslog_name setting and
     * may require that mail_task() be re-evaluated.
     */
    mail_conf_read();
    /* Enforce consistent operation of different Postfix parts. */
    import_env = mail_parm_split(VAR_IMPORT_ENVIRON, var_import_environ);
    update_env(import_env->argv);
    argv_free(import_env);
    if (tag == 0)
	tag = mail_task(argv[0]);

    /*
     * Re-initialize the logging, this time with the tag specified in main.cf
     * or on the command line.
     */
    if (isatty(STDERR_FILENO))
	msg_vstream_init(tag, VSTREAM_ERR);
    msg_syslog_init(tag, LOG_PID, LOG_FACILITY);

    /*
     * Log the command line or log lines from standard input.
     */
    if (argc > optind) {
	log_argv(level, argv + optind);
    } else {
	log_stream(level, VSTREAM_IN);
    }

    /*
     * Consistency with msg(3) functions.
     */
    if (level >= MSG_FATAL)
	sleep(1);
    exit(0);
}
