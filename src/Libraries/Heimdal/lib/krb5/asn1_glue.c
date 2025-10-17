/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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
/*
 *
 */

#include "krb5_locl.h"

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
_krb5_principal2principalname (PrincipalName *p,
			       const krb5_principal from)
{
    return copy_PrincipalName(&from->name, p);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
_krb5_principalname2krb5_principal (krb5_context context,
				    krb5_principal *principal,
				    const PrincipalName from,
				    const Realm realm)
{
    krb5_error_code ret;
    krb5_principal p;

    p = malloc(sizeof(*p));
    if (p == NULL)
	return ENOMEM;
    ret = copy_PrincipalName(&from, &p->name);
    if (ret) {
	free(p);
	return ret;
    }
    p->realm = strdup(realm);
    if (p->realm == NULL) {
	free_PrincipalName(&p->name);
        free(p);
	return ENOMEM;
    }
    *principal = p;
    return 0;
}
