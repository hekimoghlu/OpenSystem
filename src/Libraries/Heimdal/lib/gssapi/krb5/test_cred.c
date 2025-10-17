/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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
#include "gsskrb5_locl.h"
#include <err.h>
#include <getarg.h>

static void
gss_print_errors (int min_stat)
{
    OM_uint32 new_stat;
    OM_uint32 msg_ctx = 0;
    gss_buffer_desc status_string;
    OM_uint32 ret;

    do {
	ret = gss_display_status (&new_stat,
				  min_stat,
				  GSS_C_MECH_CODE,
				  GSS_C_NO_OID,
				  &msg_ctx,
				  &status_string);
	fprintf (stderr, "%.*s\n", (int)status_string.length,
				(char *)status_string.value);
	gss_release_buffer (&new_stat, &status_string);
    } while (!GSS_ERROR(ret) && msg_ctx != 0);
}

static void
gss_err(int exitval, int status, const char *fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    vwarnx (fmt, args);
    gss_print_errors (status);
    va_end(args);
    exit (exitval);
}

static void
acquire_release_loop(gss_name_t name, int counter, gss_cred_usage_t usage)
{
    OM_uint32 maj_stat, min_stat;
    gss_cred_id_t cred;
    int i;

    for (i = 0; i < counter; i++) {
	maj_stat = gss_acquire_cred(&min_stat, name,
				    GSS_C_INDEFINITE,
				    GSS_C_NO_OID_SET,
				    usage,
				    &cred,
				    NULL,
				    NULL);
	if (maj_stat != GSS_S_COMPLETE)
	    gss_err(1, min_stat, "aquire %d %d != GSS_S_COMPLETE",
		    i, (int)maj_stat);

	maj_stat = gss_release_cred(&min_stat, &cred);
	if (maj_stat != GSS_S_COMPLETE)
	    gss_err(1, min_stat, "release %d %d != GSS_S_COMPLETE",
		    i, (int)maj_stat);
    }
}


static void
acquire_add_release_add(gss_name_t name, gss_cred_usage_t usage)
{
    OM_uint32 maj_stat, min_stat;
    gss_cred_id_t cred, cred2, cred3;

    maj_stat = gss_acquire_cred(&min_stat, name,
				GSS_C_INDEFINITE,
				GSS_C_NO_OID_SET,
				usage,
				&cred,
				NULL,
				NULL);
    if (maj_stat != GSS_S_COMPLETE)
	gss_err(1, min_stat, "aquire %d != GSS_S_COMPLETE", (int)maj_stat);

    maj_stat = gss_add_cred(&min_stat,
			    cred,
			    GSS_C_NO_NAME,
			    GSS_KRB5_MECHANISM,
			    usage,
			    GSS_C_INDEFINITE,
			    GSS_C_INDEFINITE,
			    &cred2,
			    NULL,
			    NULL,
			    NULL);

    if (maj_stat != GSS_S_COMPLETE)
	gss_err(1, min_stat, "add_cred %d != GSS_S_COMPLETE", (int)maj_stat);

    maj_stat = gss_release_cred(&min_stat, &cred);
    if (maj_stat != GSS_S_COMPLETE)
	gss_err(1, min_stat, "release %d != GSS_S_COMPLETE", (int)maj_stat);

    maj_stat = gss_add_cred(&min_stat,
			    cred2,
			    GSS_C_NO_NAME,
			    GSS_KRB5_MECHANISM,
			    GSS_C_BOTH,
			    GSS_C_INDEFINITE,
			    GSS_C_INDEFINITE,
			    &cred3,
			    NULL,
			    NULL,
			    NULL);

    maj_stat = gss_release_cred(&min_stat, &cred2);
    if (maj_stat != GSS_S_COMPLETE)
	gss_err(1, min_stat, "release 2 %d != GSS_S_COMPLETE", (int)maj_stat);

    maj_stat = gss_release_cred(&min_stat, &cred3);
    if (maj_stat != GSS_S_COMPLETE)
	gss_err(1, min_stat, "release 2 %d != GSS_S_COMPLETE", (int)maj_stat);
}

static int version_flag = 0;
static int help_flag	= 0;

static struct getargs args[] = {
    {"version",	0,	arg_flag,	&version_flag, "print version", NULL },
    {"help",	0,	arg_flag,	&help_flag,  NULL, NULL }
};

static void
usage (int ret)
{
    arg_printusage (args, sizeof(args)/sizeof(*args),
		    NULL, "service@host");
    exit (ret);
}


int
main(int argc, char **argv)
{
    struct gss_buffer_desc_struct name_buffer;
    OM_uint32 maj_stat, min_stat;
    gss_name_t name;
    int optidx = 0;

    setprogname(argv[0]);
    if(getarg(args, sizeof(args) / sizeof(args[0]), argc, argv, &optidx))
	usage(1);

    if (help_flag)
	usage (0);

    if(version_flag){
	print_version(NULL);
	exit(0);
    }

    argc -= optidx;
    argv += optidx;

    if (argc < 1)
	errx(1, "argc < 1");

    name_buffer.value = argv[0];
    name_buffer.length = strlen(argv[0]);

    maj_stat = gss_import_name(&min_stat, &name_buffer,
			       GSS_C_NT_HOSTBASED_SERVICE,
			       &name);
    if (maj_stat != GSS_S_COMPLETE)
	errx(1, "import name error");

    acquire_release_loop(name, 100, GSS_C_ACCEPT);
    acquire_release_loop(name, 100, GSS_C_INITIATE);
    acquire_release_loop(name, 100, GSS_C_BOTH);

    acquire_add_release_add(name, GSS_C_ACCEPT);
    acquire_add_release_add(name, GSS_C_INITIATE);
    acquire_add_release_add(name, GSS_C_BOTH);

    gss_release_name(&min_stat, &name);

    return 0;
}
