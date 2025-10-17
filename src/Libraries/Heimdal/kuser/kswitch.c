/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 28, 2023.
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
#include "kuser_locl.h"
#include "kcc-commands.h"

#ifdef HAVE_READLINE
char *readline(const char *prompt);
#else

static char *
readline(const char *prompt)
{
    char buf[BUFSIZ];
    printf ("%s", prompt);
    fflush (stdout);
    if(fgets(buf, sizeof(buf), stdin) == NULL)
	return NULL;
    buf[strcspn(buf, "\r\n")] = '\0';
    return strdup(buf);
}

#endif

/*
 *
 */

int
kswitch(struct kswitch_options *opt, int argc, char **argv)
{
    krb5_error_code ret;
    krb5_ccache id = NULL;

    if(opt->version_flag) {
	print_version(NULL);
	exit(0);
    }

    if (opt->cache_string && opt->principal_string)
	krb5_errx(kcc_context, 1,
		  N_("Both --cache and --principal given, choose one", ""));

    if (opt->interactive_flag) {
	krb5_cc_cache_cursor cursor;
	krb5_ccache *ids = NULL;
	size_t i, len = 0;
	char *name;
	rtbl_t ct;

	ct = rtbl_create();

	rtbl_add_column_by_id(ct, 0, "#", 0);
	rtbl_add_column_by_id(ct, 1, "Principal", 0);
	rtbl_set_column_affix_by_id(ct, 1, "    ", "");
        rtbl_add_column_by_id(ct, 2, "Type", 0);
        rtbl_set_column_affix_by_id(ct, 2, "  ", "");

	ret = krb5_cc_cache_get_first(kcc_context, NULL, &cursor);
	if (ret)
	    krb5_err(kcc_context, 1, ret, "krb5_cc_cache_get_first");

	while (krb5_cc_cache_next(kcc_context, cursor, &id) == 0) {
	    krb5_principal p;
	    char num[10];

	    ret = krb5_cc_get_principal(kcc_context, id, &p);
	    if (ret)
		continue;

	    ret = krb5_unparse_name(kcc_context, p, &name);
	    krb5_free_principal(kcc_context, p);
	    if (ret)
		continue;

	    snprintf(num, sizeof(num), "%d", (int)(len + 1));
	    rtbl_add_column_entry_by_id(ct, 0, num);
	    rtbl_add_column_entry_by_id(ct, 1, name);
            rtbl_add_column_entry_by_id(ct, 2, krb5_cc_get_type(kcc_context, id));
	    free(name);

	    ids = erealloc(ids, (len + 1) * sizeof(ids[0]));
	    ids[len] = id;
	    len++;
	}
	krb5_cc_cache_end_seq_get(kcc_context, cursor);

	rtbl_format(ct, stdout);
	rtbl_destroy(ct);

	name = readline("Select number: ");
	if (name) {
	    i = atoi(name);
	    if (i == 0)
		krb5_errx(kcc_context, 1, "Cache number '%s' is invalid", name);
	    if (i > len)
		krb5_errx(kcc_context, 1, "Cache number '%s' is too large", name);

	    id = ids[i - 1];
	    ids[i - 1] = NULL;
	} else
	    krb5_errx(kcc_context, 1, "No cache selected");
	for (i = 0; i < len; i++)
	    if (ids[i])
		krb5_cc_close(kcc_context, ids[i]);

    } else if (opt->principal_string) {
	krb5_principal p;

	ret = krb5_parse_name(kcc_context, opt->principal_string, &p);
	if (ret)
	    krb5_err(kcc_context, 1, ret, "krb5_parse_name: %s",
		     opt->principal_string);

	ret = krb5_cc_cache_match(kcc_context, p, &id);
	if (ret)
	    krb5_err(kcc_context, 1, ret,
		     N_("Did not find principal: %s", ""),
		     opt->principal_string);

	krb5_free_principal(kcc_context, p);

    } else if (opt->cache_string) {
	const krb5_cc_ops *ops;
	char *str, *str2 = NULL;

	ops = krb5_cc_get_prefix_ops(kcc_context, opt->cache_string);
	if (ops == NULL) {
	    ops = krb5_cc_get_prefix_ops(kcc_context, opt->type_string);
	    if (ops == NULL)
		krb5_err(kcc_context, 1, 0, "krb5_cc_get_prefix_ops");

	    asprintf(&str2, "%s:%s", ops->prefix, opt->cache_string);
	    if (str2 == NULL)
		krb5_errx(kcc_context, 1, N_("out of memory", ""));
	    str = str2;
	} else {
	    str = opt->cache_string;
	}

	ret = krb5_cc_resolve(kcc_context, str, &id);
	if (ret)
	    krb5_err(kcc_context, 1, ret, "krb5_cc_resolve: %s", str);

	if (str2)
	    free(str2);
    } else {
	krb5_errx(kcc_context, 1, "missing option for kswitch");
    }

    ret = krb5_cc_switch(kcc_context, id);
    if (ret)
	krb5_err(kcc_context, 1, ret, "krb5_cc_switch");

    return 0;
}
