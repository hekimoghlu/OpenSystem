/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 26, 2022.
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
 * Generate subkey, from keyblock
 *
 * @param context kerberos context
 * @param key session key
 * @param etype encryption type of subkey, if ETYPE_NULL, use key's enctype
 * @param subkey returned new, free with krb5_free_keyblock().
 *
 * @return 0 on success or a Kerberos 5 error code
 *
* @ingroup krb5_crypto
 */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_generate_subkey_extended(krb5_context context,
			      const krb5_keyblock *key,
			      krb5_enctype etype,
			      krb5_keyblock **subkey)
{
    krb5_error_code ret;

    ALLOC(*subkey, 1);
    if (*subkey == NULL) {
	krb5_set_error_message(context, ENOMEM,N_("malloc: out of memory", ""));
	return ENOMEM;
    }

    if (etype == ETYPE_NULL)
	etype = key->keytype; /* use session key etype */

    /* XXX should we use the session key as input to the RF? */
    ret = krb5_generate_random_keyblock(context, etype, *subkey);
    if (ret != 0) {
	free(*subkey);
	*subkey = NULL;
    }

    return ret;
}

