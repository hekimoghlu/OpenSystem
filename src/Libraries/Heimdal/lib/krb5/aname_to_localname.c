/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_aname_to_localname (krb5_context context,
			 krb5_const_principal aname,
			 size_t lnsize,
			 char *lname)
{
    krb5_error_code ret;
    krb5_realm *lrealms, *r;
    int valid;
    size_t len;
    const char *res;

    ret = krb5_get_default_realms (context, &lrealms);
    if (ret)
	return ret;

    valid = 0;
    for (r = lrealms; *r != NULL; ++r) {
	if (strcmp (*r, aname->realm) == 0) {
	    valid = 1;
	    break;
	}
    }
    krb5_free_host_realm (context, lrealms);
    if (valid == 0)
	return KRB5_NO_LOCALNAME;

    if (aname->name.name_string.len == 1)
	res = aname->name.name_string.val[0];
    else if (aname->name.name_string.len == 2
	     && strcmp (aname->name.name_string.val[1], "root") == 0) {
	krb5_principal rootprinc;
	krb5_boolean userok;

	res = "root";

	ret = krb5_copy_principal(context, aname, &rootprinc);
	if (ret)
	    return ret;

	userok = krb5_kuserok(context, rootprinc, res);
	krb5_free_principal(context, rootprinc);
	if (!userok)
	    return KRB5_NO_LOCALNAME;

    } else
	return KRB5_NO_LOCALNAME;

    len = strlen (res);
    if (len >= lnsize)
	return ERANGE;
    strlcpy (lname, res, lnsize);

    return 0;
}
