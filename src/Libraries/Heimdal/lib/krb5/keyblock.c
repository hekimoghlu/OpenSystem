/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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
 * Zero out a keyblock
 *
 * @param keyblock keyblock to zero out
 *
 * @ingroup krb5_crypto
 */

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_keyblock_zero(krb5_keyblock *keyblock)
{
    keyblock->keytype = 0;
    krb5_data_zero(&keyblock->keyvalue);
}

/**
 * Free a keyblock's content, also zero out the content of the keyblock.
 *
 * @param context a Kerberos 5 context
 * @param keyblock keyblock content to free, NULL is valid argument
 *
 * @ingroup krb5_crypto
 */

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_free_keyblock_contents(krb5_context context,
			    krb5_keyblock *keyblock)
{
    if(keyblock) {
	if (keyblock->keyvalue.data != NULL)
	    memset(keyblock->keyvalue.data, 0, keyblock->keyvalue.length);
	krb5_data_free (&keyblock->keyvalue);
	keyblock->keytype = KRB5_ENCTYPE_NULL;
    }
}

/**
 * Free a keyblock, also zero out the content of the keyblock, uses
 * krb5_free_keyblock_contents() to free the content.
 *
 * @param context a Kerberos 5 context
 * @param keyblock keyblock to free, NULL is valid argument
 *
 * @ingroup krb5_crypto
 */

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_free_keyblock(krb5_context context,
		   krb5_keyblock *keyblock)
{
    if(keyblock){
	krb5_free_keyblock_contents(context, keyblock);
	free(keyblock);
    }
}

/**
 * Copy a keyblock, free the output keyblock with
 * krb5_free_keyblock_contents().
 *
 * @param context a Kerberos 5 context
 * @param inblock the key to copy
 * @param to the output key.
 *
 * @return 0 on success or a Kerberos 5 error code
 *
 * @ingroup krb5_crypto
 */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_copy_keyblock_contents (krb5_context context,
			     const krb5_keyblock *inblock,
			     krb5_keyblock *to)
{
    return copy_EncryptionKey(inblock, to);
}

/**
 * Copy a keyblock, free the output keyblock with
 * krb5_free_keyblock().
 *
 * @param context a Kerberos 5 context
 * @param inblock the key to copy
 * @param to the output key.
 *
 * @return 0 on success or a Kerberos 5 error code
 *
 * @ingroup krb5_crypto
 */


KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_copy_keyblock (krb5_context context,
		    const krb5_keyblock *inblock,
		    krb5_keyblock **to)
{
    krb5_error_code ret;
    krb5_keyblock *k;

    *to = NULL;

    k = calloc (1, sizeof(*k));
    if (k == NULL) {
	krb5_set_error_message(context, ENOMEM, "malloc: out of memory");
	return ENOMEM;
    }

    ret = krb5_copy_keyblock_contents (context, inblock, k);
    if (ret) {
      free(k);
      return ret;
    }
    *to = k;
    return 0;
}

/**
 * Get encryption type of a keyblock.
 *
 * @ingroup krb5_crypto
 */

KRB5_LIB_FUNCTION krb5_enctype KRB5_LIB_CALL
krb5_keyblock_get_enctype(const krb5_keyblock *block)
{
    return block->keytype;
}

/**
 * Fill in `key' with key data of type `enctype' from `data' of length
 * `size'. Key should be freed using krb5_free_keyblock_contents().
 *
 * @return 0 on success or a Kerberos 5 error code
 *
 * @ingroup krb5_crypto
 */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_keyblock_init(krb5_context context,
		   krb5_enctype type,
		   const void *data,
		   size_t size,
		   krb5_keyblock *key)
{
    krb5_error_code ret;
    size_t len;

    memset(key, 0, sizeof(*key));

    ret = krb5_enctype_keysize(context, type, &len);
    if (ret)
	return ret;

    if (len != size) {
	krb5_set_error_message(context, KRB5_PROG_ETYPE_NOSUPP,
			       "Encryption key %d is %lu bytes "
			       "long, %lu was passed in",
			       type, (unsigned long)len, (unsigned long)size);
	return KRB5_PROG_ETYPE_NOSUPP;
    }
    ret = krb5_data_copy(&key->keyvalue, data, len);
    if(ret) {
	krb5_set_error_message(context, ret, N_("malloc: out of memory", ""));
	return ret;
    }
    key->keytype = type;

    return 0;
}
