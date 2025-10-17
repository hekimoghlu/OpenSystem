/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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
#include <err.h>
#include <getarg.h>

static char *password_str = NULL;
static char *client_str = NULL;
static char *fuzzer_mode = NULL;
static int timer_mode = 0;
static int debug_flag	= 0;
static int do_lr_flag = 0;
static int version_flag = 0;
static int help_flag	= 0;

static void
tvfix(struct timeval *t1)
{
    if (t1->tv_usec < 0) {
        t1->tv_sec--;
        t1->tv_usec += 1000000;
    }
    if (t1->tv_usec >= 1000000) {
        t1->tv_sec++;
        t1->tv_usec -= 1000000;
    }
}
 
/*
 * t1 -= t2
 */

static void
tvsub(struct timeval *t1, const struct timeval *t2)
{
    t1->tv_sec  -= t2->tv_sec;
    t1->tv_usec -= t2->tv_usec;
    tvfix(t1);
}

static krb5_error_code
lr_proc(krb5_context context, krb5_last_req_entry **e, void *ctx)
{
    while (e && *e) {
	printf("e type: %d value: %d\n", (*e)->lr_type, (int)(*e)->value);
	e++;
    }
    return 0;
}

static krb5_error_code
test_get_init_creds(krb5_context context,
		    krb5_principal client)
{
    krb5_error_code ret;
    krb5_get_init_creds_opt *opt;
    krb5_creds cred;

    ret = krb5_get_init_creds_opt_alloc(context, &opt);
    if (ret)
	krb5_err(context, 1, ret, "krb5_get_init_creds_opt_alloc");

    if (do_lr_flag) {
	ret = krb5_get_init_creds_opt_set_process_last_req(context,
							   opt,
							   lr_proc,
							   NULL);
	if (ret)
	    krb5_err(context, 1, ret,
		     "krb5_get_init_creds_opt_set_process_last_req");
    }

    ret = krb5_get_init_creds_password(context,
				       &cred,
				       client,
				       password_str,
				       krb5_prompter_posix,
				       NULL,
				       0,
				       NULL,
				       opt);
    if (ret)
	krb5_warn(context, ret, "krb5_get_init_creds_password");

    krb5_get_init_creds_opt_free(context, opt);
    return ret;
}

static struct getargs args[] = {
    {"client",	0,	arg_string,	&client_str,
     "client principal to use", NULL },
    {"password",0,	arg_string,	&password_str,
     "password", NULL },
    {"fuzzer",0,	arg_string,	&fuzzer_mode,
     "turn on asn1 fuzzer, select mode", NULL },
    {"timer",0,		arg_flag,	&timer_mode,
     "timer (benchmark) mode", NULL },
    {"debug",	'd',	arg_flag,	&debug_flag,
     "turn on debuggin", NULL },
    {"last-request",	0, arg_flag,	&do_lr_flag,
     "print version", NULL },
    {"version",	0,	arg_flag,	&version_flag,
     "print version", NULL },
    {"help",	0,	arg_flag,	&help_flag,
     NULL, NULL }
};

static void
usage (int ret)
{
    arg_printusage (args, sizeof(args)/sizeof(*args), NULL, "hostname ...");
    exit (ret);
}


int
main(int argc, char **argv)
{
    krb5_context context;
    krb5_error_code ret;
    int optidx = 0, errors = 0;
    krb5_principal client;

    setprogname(argv[0]);

    if(getarg(args, sizeof(args) / sizeof(args[0]), argc, argv, &optidx))
	usage(1);

    if (help_flag)
	usage (0);

    if(version_flag){
	print_version(NULL);
	exit(0);
    }

    if(client_str == NULL)
	errx(1, "client is not set");

    ret = krb5_init_context(&context);
    if (ret)
	errx (1, "krb5_init_context failed: %d", ret);

    ret = krb5_parse_name(context, client_str, &client);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name: %d", ret);

    if (timer_mode) {
	unsigned long n;
	struct timeval tv1, tv2;

	if (password_str == NULL)
	    errx(1, "password-less mode is not support with fuzzer");

	gettimeofday(&tv1, NULL);

	for (n = 0; n < 1000; n++) {
	    if (test_get_init_creds(context, client))
		errors++;
	}
	
	gettimeofday(&tv2, NULL);

	tvsub(&tv2, &tv1);

	printf("timing: %ld.%06ld\n", 
	       (long)tv2.tv_sec, (long)tv2.tv_usec);

    } else if (fuzzer_mode) {
	unsigned long u = 0;

	if (password_str == NULL)
	    errx(1, "password-less mode is not support with fuzzer");

	if (asn1_fuzzer_method(fuzzer_mode))
	    errx(1, "no fuzzer support");

	asn1_fuzzer_reset();

	do {
	    asn1_fuzzer_next();

	    if (debug_flag)
		printf("request... %lu\n", u++);

	    /* ignore failure when we run fuzzer */
	    (void)test_get_init_creds(context, client);

	} while(!asn1_fuzzer_done());


    } else {
	if (test_get_init_creds(context, client))
	    errors++;
    }

    krb5_free_context(context);

    return errors;
}
