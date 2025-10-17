/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 6, 2023.
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
kt_remove(struct remove_options *opt, int argc, char **argv)
{
    krb5_error_code ret = 0;
    krb5_keytab_entry entry;
    krb5_keytab keytab;
    krb5_principal principal = NULL;
    krb5_enctype enctype = 0;

    if(opt->principal_string) {
	ret = krb5_parse_name(context, opt->principal_string, &principal);
	if(ret) {
	    krb5_warn(context, ret, "%s", opt->principal_string);
	    return 1;
	}
    }
    if(opt->enctype_string) {
	ret = krb5_string_to_enctype(context, opt->enctype_string, &enctype);
	if(ret) {
	    int t;
	    if(sscanf(opt->enctype_string, "%d", &t) == 1)
		enctype = t;
	    else {
		krb5_warn(context, ret, "%s", opt->enctype_string);
		if(principal)
		    krb5_free_principal(context, principal);
		return 1;
	    }
	}
    }
    if (!principal && !enctype && !opt->kvno_integer) {
	krb5_warnx(context,
		   "You must give at least one of "
		   "principal, enctype or kvno.");
	ret = EINVAL;
	goto out;
    }

    if((keytab = ktutil_open_keytab()) == NULL) {
	ret = 1;
	goto out;
    }

    entry.principal = principal;
    entry.keyblock.keytype = enctype;
    entry.vno = opt->kvno_integer;
    ret = krb5_kt_remove_entry(context, keytab, &entry);
    krb5_kt_close(context, keytab);
    if(ret)
	krb5_warn(context, ret, "remove");
 out:
    if(principal)
	krb5_free_principal(context, principal);
    return ret != 0;
}

