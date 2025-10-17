/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 19, 2025.
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
#include <asn1-common.h>
#include <heim-ipc.h>
#include <getarg.h>
#include <err.h>
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
reply(void *ctx, int errorcode, heim_idata *reply, heim_icred cred)
{
    printf("got reply\n");
    heim_ipc_semaphore_signal((heim_isemaphore)ctx); /* tell caller we are done */
}

static void
test_ipc(const char *service)
{
    heim_isemaphore s;
    heim_idata req, rep;
    heim_ipc ipc;
    int ret;

    ret = heim_ipc_init_context(service, &ipc);
    if (ret)
	errx(1, "heim_ipc_init_context: %d", ret);

    req.length = 0;
    req.data = NULL;

    ret = heim_ipc_call(ipc, &req, &rep, NULL);
    if (ret)
	errx(1, "heim_ipc_call: %d", ret);

    s = heim_ipc_semaphore_create(0);
    if (s == NULL)
	errx(1, "heim_ipc_semaphore_create");

    ret = heim_ipc_async(ipc, &req, s, reply);
    if (ret)
	errx(1, "heim_ipc_async: %d", ret);

    heim_ipc_semaphore_wait(s, HEIM_IPC_WAIT_FOREVER); /* wait for reply to complete the work */

    heim_ipc_free_context(ipc);
}


int
main(int argc, char **argv)
{
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

#ifdef __APPLE__
    test_ipc("MACH:org.h5l.test-ipc");
#endif
    test_ipc("ANY:org.h5l.test-ipc");
    test_ipc("UNIX:org.h5l.test-ipc");

    return 0;
}
