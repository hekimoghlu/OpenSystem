/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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
#include <sys/param.h>
#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <err.h>
#include <sysexits.h>

#include <smbclient/smbclient.h>
#include <smbclient/ntstatus.h>


#include "common.h"

extern char *__progname;

static void help(void);
static void help_usage(void);
static int  cmd_help(int argc, char *argv[]);

int verbose = 0;

typedef int cmd_fn_t (int argc, char *argv[]);
typedef void cmd_usage_t (void);


static struct commands {
	const char *	name;
	cmd_fn_t*	fn;
	cmd_usage_t *	usage;
} commands[] = {
	{"help",		cmd_help,		help_usage},
	{"lookup",		cmd_lookup,		lookup_usage},
	{"status",		cmd_status,		status_usage},
	{"view",		cmd_view,		view_usage},
	{"dfs",			cmd_dfs,		dfs_usage},
	{"identity",	cmd_identity,	identity_usage},
    {"statshares",  cmd_statshares, statshares_usage},
    {"multichannel", cmd_multichannel, multichannel_usage},
    {"snapshot",    cmd_snapshot,   snapshot_usage},
    {"smbstat",     cmd_smbstat,    smbstat_usage},
	{NULL, NULL, NULL}
};

static struct commands *
lookupcmd(const char *name)
{
	struct commands *cmd;

	for (cmd = commands; cmd->name; cmd++) {
		if (strcmp(cmd->name, name) == 0)
			return cmd;
	}
	return NULL;
}

void 
ntstatus_to_err(NTSTATUS status)
{
	switch (status) {
		case STATUS_NO_SUCH_DEVICE:
			err(EX_UNAVAILABLE, "failed to intitialize the smb library");
			break;
		case STATUS_LOGON_FAILURE:
			err(EX_NOPERM, "server rejected the authentication");
			break;
		case STATUS_CONNECTION_REFUSED:
			err(EX_NOHOST, "server connection failed");
			break;
		case STATUS_NO_SUCH_USER:
			err(EX_NOUSER, "no such network user");
			break;
		case STATUS_INVALID_HANDLE:
			err(EX_UNAVAILABLE, "invalid handle, internal error");
			break;
		case STATUS_NO_MEMORY:
			err(EX_UNAVAILABLE, "no memory, internal error");
			break;
		case STATUS_INVALID_PARAMETER:
			err(EX_USAGE, "Invalid parameter. Please correct the URL/Path and try again");
			break;
		case STATUS_BAD_NETWORK_NAME:
			err(EX_NOHOST, "share name doesn't exist");
			break;
		case STATUS_NOT_SUPPORTED:
			err(EX_NOHOST, "operation not supported by server");
			break;
		default:
			err(EX_OSERR, "unknown status %d", status);
			break;
	}
}

int
cmd_help(int argc, char *argv[])
{
	struct commands *cmd;
	char *cp;
    
	if (argc < 2)
		help_usage();
	cp = argv[1];
	cmd = lookupcmd(cp);
	if (cmd == NULL)
		errx(EX_DATAERR, "unknown command %s", cp);
	if (cmd->usage == NULL)
		errx(EX_DATAERR, "no specific help for command %s", cp);
	cmd->usage();
	exit(0);
}

int
main(int argc, char *argv[])
{
	struct commands *cmd;
	char *cp;
	int opt;

	if (argc < 2)
		help();

	while ((opt = getopt(argc, argv, "hv")) != EOF) {
		switch (opt) {
		    case 'h':
			help();
			/*NOTREACHED */
		    case 'v':
			verbose = 1;
			break;
		    default:
			warnx("invalid option %c", opt);
			help();
			/*NOTREACHED */
		}
	}
	if (optind >= argc)
		help();

	cp = argv[optind];
	cmd = lookupcmd(cp);
	if (cmd == NULL)
		errx(EX_DATAERR, "unknown command %s", cp);

	argc -= optind;
	argv += optind;
	optind = optreset = 1;
	return cmd->fn(argc, argv);
}

static void
help(void) {
	fprintf(stderr, "\n");
	fprintf(stderr, "usage: %s [-hv] subcommand [args]\n", __progname);
	fprintf(stderr, "where subcommands are:\n"
	" help          display help on specified subcommand\n"
	" lookup        resolve NetBIOS name to IP address\n"
	" status        resolve IP address or DNS name to NetBIOS names\n"
	" view          list resources on specified host\n"
	" dfs           list DFS referrals\n"
	" identity      identity of the user as known by the specified host\n"
    " statshares    list the attributes of mounted share(s)\n"
    " multichannel  list the attributes of the channels of mounted share(s)\n"
    " snapshot      list snapshots for the mount path \n"
    " smbstat       list info about item at path \n"
	"\n");
	exit(1);
}

static void
help_usage(void) {
	fprintf(stderr, "usage: smbutil help command\n");
	exit(1);
}
