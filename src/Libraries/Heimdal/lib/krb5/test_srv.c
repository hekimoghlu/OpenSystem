/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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
#include "krb5_locl.h"
#include <getarg.h>

/*
 *
 */

static int version_flag = 0;
static int help_flag	= 0;

static struct getargs args[] = {
    {"version",		0,	arg_flag,	&version_flag,
     "print version", NULL },
    {"help",		0,	arg_flag,	&help_flag,
     NULL, NULL }
};

static void
usage(int ret)
{
    arg_printusage(args,
		   sizeof(args)/sizeof(*args),
		   NULL,
		   "");
    exit(ret);
}

#define NUM_ITER 100000
#define MAX_HOSTS 1000

int
main(int argc, char **argv)
{
    struct _krb5_srv_query_ctx ctx;
    krb5_context context;
    krb5_error_code ret;
    int optidx = 0;
    size_t n, m;

    setprogname(argv[0]);

    if(getarg(args, sizeof(args) / sizeof(args[0]), argc, argv, &optidx))
	usage(1);

    if (help_flag)
	usage (0);

    if(version_flag){
	print_version(NULL);
	exit(0);
    }

    ret = krb5_init_context (&context);
    if (ret)
	errx (1, "krb5_init_context failed: %d", ret);

    memset(&ctx, 0, sizeof(ctx));

    ctx.context = context;
    ctx.domain = rk_UNCONST("domain");
    ctx.array = calloc(MAX_HOSTS, sizeof(ctx.array[0]));
    if (ctx.array == NULL)
	errx(1, "malloc: outo of memory");

#ifdef __APPLE__
    for (n = 0; n < NUM_ITER; n++) {

	if ((n % (NUM_ITER / 10)) == 0) {
	    printf("%d ", (int)n);
	    fflush(stdout);
	}

	if (n < 10) {
	    ctx.len = n;
	} else {
	    ctx.len = (rk_random() % (MAX_HOSTS - 1)) + 1;
	}

	for (m = 0; m < ctx.len; m++) {
	    ctx.array[m] = calloc(1, sizeof(ctx.array[m][0]));
	    ctx.array[m]->priority = rk_random_uniform(5);
	    ctx.array[m]->weight = rk_random_uniform(4);
	}

	_krb5_state_srv_sort(&ctx);
    }
#endif
    printf("\n");


    krb5_free_context(context);

    return 0;
}
