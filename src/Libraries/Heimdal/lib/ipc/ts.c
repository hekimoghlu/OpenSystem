/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 27, 2025.
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
#include <stdio.h>
#include <stdlib.h>
#include <krb5-types.h>
#include <heim-ipc.h>
#include <getarg.h>
#include <roken.h>

static int help_flag;
static int version_flag;

static struct getargs args[] = {
    {	"help",		'h',	arg_flag,   &help_flag },
    {	"version",	'v',	arg_flag,   &version_flag }
};

static int num_args = sizeof(args) / sizeof(args[0]);

static void
usage(int ret)
{
    arg_printusage (args, num_args, NULL, "");
    exit (ret);
}

static void
test_service(void *ctx, const heim_idata *req,
	     const heim_icred cred,
	     heim_ipc_complete complete,
	     heim_sipc_call cctx)
{
    heim_idata rep;
    printf("got request\n");
    rep.length = 0;
    rep.data = NULL;
    (*complete)(cctx, 0, &rep);
}


int
main(int argc, char **argv)
{
    heim_sipc u;
    int optidx = 0;

    setprogname(argv[0]);

    if (getarg(args, num_args, argc, argv, &optidx))
	usage(1);

    if (help_flag)
	usage(0);

    if (version_flag) {
	print_version(NULL);
	exit(0);
    }

#if __APPLE__
    {
	heim_sipc mach;
	heim_sipc_launchd_mach_init("org.h5l.test-ipc",
				    test_service, NULL, &mach);
    }
#endif
    heim_sipc_service_unix("org.h5l.test-ipc",
			   test_service, NULL, &u);
    heim_ipc_main();
}
