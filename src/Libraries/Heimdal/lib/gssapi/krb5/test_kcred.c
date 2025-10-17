/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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

static int version_flag = 0;
static int help_flag	= 0;

static void
copy_import(void)
{
    gss_cred_id_t cred1, cred2;
    OM_uint32 maj_stat, min_stat;
    gss_name_t name1, name2;
    OM_uint32 lifetime1, lifetime2;
    gss_cred_usage_t usage1, usage2;
    gss_OID_set mechs1, mechs2;
    krb5_ccache id;
    krb5_error_code ret;
    krb5_context context;
    int equal;

    maj_stat = gss_acquire_cred(&min_stat, GSS_C_NO_NAME, GSS_C_INDEFINITE,
				GSS_C_NO_OID_SET, GSS_C_INITIATE,
				&cred1, NULL, NULL);
    if (maj_stat != GSS_S_COMPLETE)
	errx(1, "gss_acquire_cred");

    maj_stat = gss_inquire_cred(&min_stat, cred1, &name1, &lifetime1,
				&usage1, &mechs1);
    if (maj_stat != GSS_S_COMPLETE)
	errx(1, "gss_inquire_cred");

    ret = krb5_init_context(&context);
    if (ret)
	errx(1, "krb5_init_context");

    ret = krb5_cc_new_unique(context, krb5_cc_type_memory, NULL, &id);
    if (ret)
	krb5_err(context, 1, ret, "krb5_cc_new_unique");

    maj_stat = gss_krb5_copy_ccache(&min_stat, context, cred1, id);
    if (maj_stat != GSS_S_COMPLETE)
	errx(1, "gss_krb5_copy_ccache");

    maj_stat = gss_krb5_import_cred(&min_stat, id, NULL, NULL, &cred2);
    if (maj_stat != GSS_S_COMPLETE)
	errx(1, "gss_krb5_import_cred");

    maj_stat = gss_inquire_cred(&min_stat, cred2, &name2, &lifetime2,
				&usage2, &mechs2);
    if (maj_stat != GSS_S_COMPLETE)
	errx(1, "gss_inquire_cred 2");

    maj_stat = gss_compare_name(&min_stat, name1, name2, &equal);
    if (maj_stat != GSS_S_COMPLETE)
	errx(1, "gss_compare_name");
    if (!equal)
	errx(1, "names not equal");

    if (lifetime1 != lifetime1)
	errx(1, "lifetime not equal");

    if (usage1 != usage1)
	errx(1, "usage not equal");

    gss_release_cred(&min_stat, &cred1);
    gss_release_cred(&min_stat, &cred2);

    gss_release_name(&min_stat, &name1);
    gss_release_name(&min_stat, &name2);

#if 0
    compare(mechs1, mechs2);
#endif

    gss_release_oid_set(&min_stat, &mechs1);
    gss_release_oid_set(&min_stat, &mechs2);

    krb5_cc_destroy(context, id);
    krb5_free_context(context);
}

static struct getargs args[] = {
    {"version",	0,	arg_flag,	&version_flag, "print version", NULL },
    {"help",	0,	arg_flag,	&help_flag,  NULL, NULL }
};

static void
usage (int ret)
{
    arg_printusage (args, sizeof(args)/sizeof(*args),
		    NULL, "");
    exit (ret);
}

int
main(int argc, char **argv)
{
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

    copy_import();

    return 0;
}
