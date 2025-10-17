/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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
#include "ktutil_locl.h"
#include <err.h>

static void usage(int status) __attribute__((noreturn));


static int help_flag;
static int version_flag;
int verbose_flag;
char *keytab_string;
static char keytab_buf[256];

static struct getargs args[] = {
    {
	"version",
	0,
	arg_flag,
	&version_flag,
	NULL,
	NULL
    },
    {
	"help",
	'h',
	arg_flag,
	&help_flag,
	NULL,
	NULL
    },
    {
	"keytab",
	'k',
	arg_string,
	&keytab_string,
	"keytab",
	"keytab to operate on"
    },
    {
	"verbose",
	'v',
	arg_flag,
	&verbose_flag,
	"verbose",
	"run verbosely"
    }
};

static int num_args = sizeof(args) / sizeof(args[0]);

krb5_context context;

krb5_keytab
ktutil_open_keytab(void)
{
    krb5_error_code ret;
    krb5_keytab keytab;
    if (keytab_string == NULL) {
	ret = krb5_kt_default_name (context, keytab_buf, sizeof(keytab_buf));
	if (ret) {
	    krb5_warn(context, ret, "krb5_kt_default_name");
	    return NULL;
	}
	keytab_string = keytab_buf;
    }
    ret = krb5_kt_resolve(context, keytab_string, &keytab);
    if (ret) {
	krb5_warn(context, ret, "resolving keytab %s", keytab_string);
	return NULL;
    }
    if (verbose_flag)
	fprintf (stderr, "Using keytab %s\n", keytab_string);

    return keytab;
}

int
help(void *opt, int argc, char **argv)
{
    if(argc == 0) {
	sl_help(commands, 1, argv - 1 /* XXX */);
    } else {
	SL_cmd *c = sl_match (commands, argv[0], 0);
 	if(c == NULL) {
	    fprintf (stderr, "No such command: %s. "
		     "Try \"help\" for a list of commands\n",
		     argv[0]);
	} else {
	    if(c->func) {
		char shelp[] = "--help";
		char *fake[3];
		fake[0] = argv[0];
		fake[1] = shelp;
		fake[2] = NULL;
		(*c->func)(2, fake);
		fprintf(stderr, "\n");
	    }
	    if(c->help && *c->help)
		fprintf (stderr, "%s\n", c->help);
	    if((++c)->name && c->func == NULL) {
		int f = 0;
		fprintf (stderr, "Synonyms:");
		while (c->name && c->func == NULL) {
		    fprintf (stderr, "%s%s", f ? ", " : " ", (c++)->name);
		    f = 1;
		}
		fprintf (stderr, "\n");
	    }
	}
    }
    return 0;
}

static void
usage(int status)
{
    arg_printusage(args, num_args, NULL, "command");
    exit(status);
}

int
main(int argc, char **argv)
{
    int optidx = 0;
    krb5_error_code ret;
    setprogname(argv[0]);
    ret = krb5_init_context(&context);
    if (ret)
	errx (1, "krb5_init_context failed: %d", ret);
    if(getarg(args, num_args, argc, argv, &optidx))
	usage(1);
    if(help_flag)
	usage(0);
    if(version_flag) {
	print_version(NULL);
	exit(0);
    }
    argc -= optidx;
    argv += optidx;
    if(argc == 0)
	usage(1);
    ret = sl_command(commands, argc, argv);
    if(ret == -1)
	krb5_warnx (context, "unrecognized command: %s", argv[0]);
    return ret;
}
