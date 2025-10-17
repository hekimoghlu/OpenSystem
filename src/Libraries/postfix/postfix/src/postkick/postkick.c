/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 19, 2024.
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
#include <fcntl.h>
#include <unistd.h>
#include <syslog.h>
#include <string.h>
#include <stdlib.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <vstream.h>
#include <msg_vstream.h>
#include <safe.h>
#include <events.h>
#include <warn_stat.h>
#include <clean_env.h>

/* Global library. */

#include <mail_proto.h>
#include <mail_params.h>
#include <mail_version.h>
#include <mail_conf.h>
#include <mail_parm_split.h>

static NORETURN usage(char *myname)
{
    msg_fatal("usage: %s [-c config_dir] [-v] class service request", myname);
}

MAIL_VERSION_STAMP_DECLARE;

int     main(int argc, char **argv)
{
    char   *class;
    char   *service;
    char   *request;
    int     fd;
    struct stat st;
    char   *slash;
    int     c;
    ARGV   *import_env;

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

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
     * Process environment options as early as we can.
     */
    if (safe_getenv(CONF_ENV_VERB))
	msg_verbose = 1;

    /*
     * Initialize. Set up logging, read the global configuration file and
     * extract configuration information.
     */
    if ((slash = strrchr(argv[0], '/')) != 0 && slash[1])
	argv[0] = slash + 1;
    msg_vstream_init(argv[0], VSTREAM_ERR);
    set_mail_conf_str(VAR_PROCNAME, var_procname = mystrdup(argv[0]));

    /*
     * Parse JCL.
     */
    while ((c = GETOPT(argc, argv, "c:v")) > 0) {
	switch (c) {
	default:
	    usage(argv[0]);
	case 'c':
	    if (setenv(CONF_ENV_PATH, optarg, 1) < 0)
		msg_fatal("out of memory");
	    break;
	case 'v':
	    msg_verbose++;
	    break;
	}
    }
    if (argc != optind + 3)
	usage(argv[0]);
    class = argv[optind];
    service = argv[optind + 1];
    request = argv[optind + 2];

    /*
     * Finish initializations.
     */
    mail_conf_read();
    /* Enforce consistent operation of different Postfix parts. */
    import_env = mail_parm_split(VAR_IMPORT_ENVIRON, var_import_environ);
    update_env(import_env->argv);
    argv_free(import_env);
    if (chdir(var_queue_dir))
	msg_fatal("chdir %s: %m", var_queue_dir);

    /*
     * Kick the service.
     */
    if (mail_trigger(class, service, request, strlen(request)) < 0) {
	msg_warn("Cannot contact class %s service %s - perhaps the mail system is down",
		 class, service);
	exit(1);
    }

    /*
     * Problem: With triggers over full duplex (i.e. non-FIFO) channels, we
     * must avoid closing the channel before the server has received the
     * request. Otherwise some hostile kernel may throw away the request.
     * 
     * Solution: The trigger routine registers a read event handler that runs
     * when the server closes the channel. The event_drain() routine waits
     * for the event handler to run, but gives up when it takes too long.
     */
    else {
	event_drain(var_event_drain);
	exit(0);
    }
}
