/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
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

/**
 * Free all memory allocated by `realmlist'
 *
 * @param context A Kerberos 5 context.
 * @param realmlist realmlist to free, NULL is ok
 *
 * @return a Kerberos error code, always 0.
 *
 * @ingroup krb5_support
 */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_free_host_realm(krb5_context context,
		     krb5_realm *realmlist)
{
    krb5_realm *p;

    if(realmlist == NULL)
	return 0;
    for (p = realmlist; *p; ++p)
	free (*p);
    free (realmlist);
    return 0;
}
