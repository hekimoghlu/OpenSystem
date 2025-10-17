/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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

RCSID("$Id$");


static krb5_boolean
compare_keyblock(const krb5_keyblock *a, const krb5_keyblock *b)
{
    if(a->keytype != b->keytype ||
       a->keyvalue.length != b->keyvalue.length ||
       memcmp(a->keyvalue.data, b->keyvalue.data, a->keyvalue.length) != 0)
	return FALSE;
    return TRUE;
}

int
kt_copy (struct copy_options *opt, int argc, char **argv)
{
    krb5_keytab src_keytab = NULL, dst_keytab = NULL;
    krb5_principal match_principal = NULL;
    krb5_keytab_entry entry, dummy;
    const char *from = argv[0];
    const char *to = argv[1];
    krb5_kt_cursor cursor;
    krb5_error_code ret;

    ret = krb5_kt_resolve (context, from, &src_keytab);
    if (ret) {
	krb5_warn (context, ret, "resolving src keytab `%s'", from);
	goto out;
    }

    ret = krb5_kt_resolve (context, to, &dst_keytab);
    if (ret) {
	krb5_warn (context, ret, "resolving dst keytab `%s'", to);
	goto out;
    }

    if (opt->match_principal_string) {
	ret = krb5_parse_name(context,
			      opt->match_principal_string,
			      &match_principal);
	if (ret) {
	    krb5_warn (context, ret, "failed parsing match principal `%s'",
		       opt->match_principal_string);
	    goto out;
	}
	if (verbose_flag) {
	    char *str = NULL;
	    ret = krb5_unparse_name(context, match_principal, &str);
	    if (ret == 0) {
		fprintf(stderr, "matching on principal %s\n", str);
		krb5_xfree(str);
	    }
	}
    }

    ret = krb5_kt_start_seq_get (context, src_keytab, &cursor);
    if (ret) {
	krb5_warn (context, ret, "krb5_kt_start_seq_get %s", keytab_string);
	goto out;
    }

    if (verbose_flag)
	fprintf(stderr, "copying %s to %s\n", from, to);

    while((ret = krb5_kt_next_entry(context, src_keytab,
				    &entry, &cursor)) == 0) {
	char *name_str;
	char *etype_str;
	ret = krb5_unparse_name (context, entry.principal, &name_str);
	if(ret) {
	    krb5_warn(context, ret, "krb5_unparse_name");
	    name_str = NULL; /* XXX */
	}
	ret = krb5_enctype_to_string(context, entry.keyblock.keytype, &etype_str);
	if(ret) {
	    krb5_warn(context, ret, "krb5_enctype_to_string");
	    etype_str = NULL; /* XXX */
	}

	if (match_principal &&
	    !krb5_principal_match(context, entry.principal, match_principal))
	{
	    if (verbose_flag) {
		krb5_warnx(context, "skipping %s, keytype %s, kvno %d",
			   name_str, etype_str, entry.vno);
	    }
	    free(name_str);
	    free(etype_str);
	    continue;
	}

	ret = krb5_kt_get_entry(context, dst_keytab,
				entry.principal,
				entry.vno,
				entry.keyblock.keytype,
				&dummy);
	if(ret == 0) {
	    /* this entry is already in the new keytab, so no need to
               copy it; if the keyblocks are not the same, something
               is weird, so complain about that */
	    if(!compare_keyblock(&entry.keyblock, &dummy.keyblock)) {
		krb5_warnx(context, "entry with different keyvalue "
			   "already exists for %s, keytype %s, kvno %d",
			   name_str, etype_str, entry.vno);
	    }
	    krb5_kt_free_entry(context, &dummy);
	    krb5_kt_free_entry (context, &entry);
	    free(name_str);
	    free(etype_str);
	    continue;
	} else if(ret != KRB5_KT_NOTFOUND) {
	    krb5_warn (context, ret, "%s: fetching %s/%s/%u",
		       to, name_str, etype_str, entry.vno);
	    krb5_kt_free_entry (context, &entry);
	    free(name_str);
	    free(etype_str);
	    break;
	}
	if (verbose_flag)
	    fprintf (stderr, "copying %s, keytype %s, kvno %d\n", name_str,
		     etype_str, entry.vno);
	ret = krb5_kt_add_entry (context, dst_keytab, &entry);
	krb5_kt_free_entry (context, &entry);
	if (ret) {
	    krb5_warn (context, ret, "%s: adding %s/%s/%u",
		       to, name_str, etype_str, entry.vno);
	    free(name_str);
	    free(etype_str);
	    break;
	}
	free(name_str);
	free(etype_str);
    }
    krb5_kt_end_seq_get (context, src_keytab, &cursor);

  out:
    if (match_principal)
	krb5_free_principal(context, match_principal);
    if (src_keytab)
	krb5_kt_close (context, src_keytab);
    if (dst_keytab)
	krb5_kt_close (context, dst_keytab);
    return ret != 0;
}
