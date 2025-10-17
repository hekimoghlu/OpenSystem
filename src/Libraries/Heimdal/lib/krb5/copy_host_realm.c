/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 13, 2021.
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
 * Copy the list of realms from `from' to `to'.
 *
 * @param context Kerberos 5 context.
 * @param from list of realms to copy from.
 * @param to list of realms to copy to, free list of krb5_free_host_realm().
 *
 * @return Returns 0 to indicate success. Otherwise an kerberos et
 * error code is returned, see krb5_get_error_message().
 *
 * @ingroup krb5
 */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_copy_host_realm(krb5_context context,
		     const krb5_realm *from,
		     krb5_realm **to)
{
    unsigned int n, i;
    const krb5_realm *p;

    for (n = 1, p = from; *p != NULL; ++p)
	++n;

    *to = calloc (n, sizeof(**to));
    if (*to == NULL) {
	krb5_set_error_message (context, ENOMEM,
				N_("malloc: out of memory", ""));
	return ENOMEM;
    }

    for (i = 0, p = from; *p != NULL; ++p, ++i) {
	(*to)[i] = strdup(*p);
	if ((*to)[i] == NULL) {
	    krb5_free_host_realm (context, *to);
	    krb5_set_error_message (context, ENOMEM,
				    N_("malloc: out of memory", ""));
	    return ENOMEM;
	}
    }
    return 0;
}
