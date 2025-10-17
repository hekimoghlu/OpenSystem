/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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
#include <sys/wait.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <pwd.h>
#include <grp.h>
#include <fcntl.h>
#ifdef STRCASECMP_IN_STRINGS_H
#include <strings.h>
#endif

/* Utility library. */

#include <msg.h>
#include <argv.h>
#include <dict.h>
#include <mymalloc.h>
#include <spawn_command.h>
#include <split_at.h>
#include <timed_wait.h>
#include <set_eugid.h>

/* Global library. */

#include <mail_version.h>

/* Single server skeleton. */

#include <mail_params.h>
#include <mail_server.h>
#include <mail_conf.h>
#include <mail_parm_split.h>

/* Application-specific. */

 /*
  * Tunable parameters. Values are taken from the config file, after
  * prepending the service name to _name, and so on.
  */
int     var_command_maxtime;		/* system-wide */

 /*
  * For convenience. Instead of passing around lists of parameters, bundle
  * them up in convenient structures.
  */
typedef struct {
    char  **argv;			/* argument vector */
    uid_t   uid;			/* command privileges */
    gid_t   gid;			/* command privileges */
    int     time_limit;			/* per-service time limit */
} SPAWN_ATTR;

/* get_service_attr - get service attributes */

static void get_service_attr(SPAWN_ATTR *attr, char *service, char **argv)
{
    const char *myname = "get_service_attr";
    struct passwd *pwd;
    struct group *grp;
    char   *user;			/* user name */
    char   *group;			/* group name */

    /*
     * Initialize.
     */
    user = 0;
    group = 0;
    attr->argv = 0;

    /*
     * Figure out the command time limit for this transport.
     */
    attr->time_limit =
	get_mail_conf_time2(service, _MAXTIME, var_command_maxtime, 's', 1, 0);

    /*
     * Iterate over the command-line attribute list.
     */
    for ( /* void */ ; *argv != 0; argv++) {

	/*
	 * user=username[:groupname]
	 */
	if (strncasecmp("user=", *argv, sizeof("user=") - 1) == 0) {
	    user = *argv + sizeof("user=") - 1;
	    if ((group = split_at(user, ':')) != 0)	/* XXX clobbers argv */
		if (*group == 0)
		    group = 0;
	    if ((pwd = getpwnam(user)) == 0)
		msg_fatal("unknown user name: %s", user);
	    attr->uid = pwd->pw_uid;
	    if (group != 0) {
		if ((grp = getgrnam(group)) == 0)
		    msg_fatal("unknown group name: %s", group);
		attr->gid = grp->gr_gid;
	    } else {
		attr->gid = pwd->pw_gid;
	    }
	}

	/*
	 * argv=command...
	 */
	else if (strncasecmp("argv=", *argv, sizeof("argv=") - 1) == 0) {
	    *argv += sizeof("argv=") - 1;	/* XXX clobbers argv */
	    attr->argv = argv;
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
	msg_fatal("missing user= attribute");
    if (attr->argv == 0)
	msg_fatal("missing argv= attribute");
    if (attr->uid == 0)
	msg_fatal("request to deliver as root");
    if (attr->uid == var_owner_uid)
	msg_fatal("request to deliver as mail system owner");
    if (attr->gid == 0)
	msg_fatal("request to use privileged group id %ld", (long) attr->gid);
    if (attr->gid == var_owner_gid)
	msg_fatal("request to use mail system owner group id %ld", (long) attr->gid);
    if (attr->uid == (uid_t) (-1))
	msg_fatal("user must not have user ID -1");
    if (attr->gid == (gid_t) (-1))
	msg_fatal("user must not have group ID -1");

    /*
     * Give the poor tester a clue of what is going on.
     */
    if (msg_verbose)
	msg_info("%s: uid %ld, gid %ld; time %d",
	      myname, (long) attr->uid, (long) attr->gid, attr->time_limit);
}

/* spawn_service - perform service for client */

static void spawn_service(VSTREAM *client_stream, char *service, char **argv)
{
    const char *myname = "spawn_service";
    static SPAWN_ATTR attr;
    WAIT_STATUS_T status;
    ARGV   *export_env;

    /*
     * This routine runs whenever a client connects to the UNIX-domain socket
     * dedicated to running an external command.
     */
    if (msg_verbose)
	msg_info("%s: service=%s, command=%s...", myname, service, argv[0]);

    /*
     * Look up service attributes and config information only once. This is
     * safe since the information comes from a trusted source.
     */
    if (attr.argv == 0) {
	get_service_attr(&attr, service, argv);
    }

    /*
     * Execute the command.
     */
    export_env = mail_parm_split(VAR_EXPORT_ENVIRON, var_export_environ);
    status = spawn_command(CA_SPAWN_CMD_STDIN(vstream_fileno(client_stream)),
			 CA_SPAWN_CMD_STDOUT(vstream_fileno(client_stream)),
			 CA_SPAWN_CMD_STDERR(vstream_fileno(client_stream)),
			   CA_SPAWN_CMD_UID(attr.uid),
			   CA_SPAWN_CMD_GID(attr.gid),
			   CA_SPAWN_CMD_ARGV(attr.argv),
			   CA_SPAWN_CMD_TIME_LIMIT(attr.time_limit),
			   CA_SPAWN_CMD_EXPORT(export_env->argv),
			   CA_SPAWN_CMD_END);
    argv_free(export_env);

    /*
     * Warn about unsuccessful completion.
     */
    if (!NORMAL_EXIT_STATUS(status)) {
	if (WIFEXITED(status))
	    msg_warn("command %s exit status %d",
		     attr.argv[0], WEXITSTATUS(status));
	if (WIFSIGNALED(status))
	    msg_warn("command %s killed by signal %d",
		     attr.argv[0], WTERMSIG(status));
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

MAIL_VERSION_STAMP_DECLARE;

/* main - pass control to the single-threaded skeleton */

int     main(int argc, char **argv)
{
    static const CONFIG_TIME_TABLE time_table[] = {
	VAR_COMMAND_MAXTIME, DEF_COMMAND_MAXTIME, &var_command_maxtime, 1, 0,
	0,
    };

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    single_server_main(argc, argv, spawn_service,
		       CA_MAIL_SERVER_TIME_TABLE(time_table),
		       CA_MAIL_SERVER_POST_INIT(drop_privileges),
		       CA_MAIL_SERVER_PRE_ACCEPT(pre_accept),
		       CA_MAIL_SERVER_PRIVILEGED,
		       0);
}
