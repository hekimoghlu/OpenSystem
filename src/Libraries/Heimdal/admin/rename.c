/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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

int
kt_rename(struct rename_options *opt, int argc, char **argv)
{
    krb5_error_code ret = 0;
    krb5_keytab_entry entry;
    krb5_keytab keytab;
    krb5_kt_cursor cursor;
    krb5_principal from_princ, to_princ;

    ret = krb5_parse_name(context, argv[0], &from_princ);
    if(ret != 0) {
	krb5_warn(context, ret, "%s", argv[0]);
	return 1;
    }

    ret = krb5_parse_name(context, argv[1], &to_princ);
    if(ret != 0) {
	krb5_free_principal(context, from_princ);
	krb5_warn(context, ret, "%s", argv[1]);
	return 1;
    }

    if((keytab = ktutil_open_keytab()) == NULL) {
	krb5_free_principal(context, from_princ);
	krb5_free_principal(context, to_princ);
	return 1;
    }

    ret = krb5_kt_start_seq_get(context, keytab, &cursor);
    if(ret) {
	krb5_kt_close(context, keytab);
	krb5_free_principal(context, from_princ);
	krb5_free_principal(context, to_princ);
	return 1;
    }
    while(1) {
	ret = krb5_kt_next_entry(context, keytab, &entry, &cursor);
	if(ret != 0) {
	    if(ret != KRB5_CC_END && ret != KRB5_KT_END)
		krb5_warn(context, ret, "getting entry from keytab");
	    else
		ret = 0;
	    break;
	}
	if(krb5_principal_compare(context, entry.principal, from_princ)) {
	    krb5_free_principal(context, entry.principal);
	    entry.principal = to_princ;
	    ret = krb5_kt_add_entry(context, keytab, &entry);
	    if(ret) {
		entry.principal = NULL;
		krb5_kt_free_entry(context, &entry);
		krb5_warn(context, ret, "adding entry");
		break;
	    }
	    if (opt->delete_flag) {
		entry.principal = from_princ;
		ret = krb5_kt_remove_entry(context, keytab, &entry);
		if(ret) {
		    entry.principal = NULL;
		    krb5_kt_free_entry(context, &entry);
		    krb5_warn(context, ret, "removing entry");
		    break;
		}
	    }
	    entry.principal = NULL;
	}
	krb5_kt_free_entry(context, &entry);
    }
    krb5_kt_end_seq_get(context, keytab, &cursor);

    krb5_free_principal(context, from_princ);
    krb5_free_principal(context, to_princ);

    return ret != 0;
}

