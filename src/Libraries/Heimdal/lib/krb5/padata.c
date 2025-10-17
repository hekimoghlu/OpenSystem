/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 21, 2024.
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

KRB5_LIB_FUNCTION PA_DATA * KRB5_LIB_CALL
krb5_find_padata(PA_DATA *val, unsigned len, int type, int *idx)
{
    for(; *idx < (int)len; (*idx)++)
	if(val[*idx].padata_type == (unsigned)type)
	    return val + *idx;
    return NULL;
}

KRB5_LIB_FUNCTION int KRB5_LIB_CALL
krb5_padata_add(krb5_context context, METHOD_DATA *md,
		int type, void *buf, size_t len)
{
    PA_DATA *pa;

    pa = realloc (md->val, (md->len + 1) * sizeof(*md->val));
    if (pa == NULL) {
	krb5_set_error_message(context, ENOMEM,
			       N_("malloc: out of memory", ""));
	return ENOMEM;
    }
    md->val = pa;

    pa[md->len].padata_type = type;
    pa[md->len].padata_value.length = len;
    pa[md->len].padata_value.data = buf;
    md->len++;

    return 0;
}
