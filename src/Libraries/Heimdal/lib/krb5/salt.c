/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 6, 2022.
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

/* coverity[+alloc : arg-*3] */
KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_salttype_to_string (krb5_context context,
			 krb5_enctype etype,
			 krb5_salttype stype,
			 char **string)
{
    struct _krb5_encryption_type *e;
    struct salt_type *st;

    e = _krb5_find_enctype (etype);
    if (e == NULL) {
	krb5_set_error_message(context, KRB5_PROG_ETYPE_NOSUPP,
			       "encryption type %d not supported",
			       etype);
	return KRB5_PROG_ETYPE_NOSUPP;
    }
    for (st = e->keytype->string_to_key; st && st->type; st++) {
	if (st->type == stype) {
	    *string = strdup (st->name);
	    if (*string == NULL) {
		krb5_set_error_message (context, ENOMEM,
					N_("malloc: out of memory", ""));
		return ENOMEM;
	    }
	    return 0;
	}
    }
    krb5_set_error_message (context, HEIM_ERR_SALTTYPE_NOSUPP,
			    "salttype %u not supported", stype);
    return HEIM_ERR_SALTTYPE_NOSUPP;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_string_to_salttype (krb5_context context,
			 krb5_enctype etype,
			 const char *string,
			 krb5_salttype *salttype)
{
    struct _krb5_encryption_type *e;
    struct salt_type *st;

    e = _krb5_find_enctype (etype);
    if (e == NULL) {
	krb5_set_error_message(context, KRB5_PROG_ETYPE_NOSUPP,
			       N_("encryption type %d not supported", ""),
			       etype);
	return KRB5_PROG_ETYPE_NOSUPP;
    }
    for (st = e->keytype->string_to_key; st && st->type; st++) {
	if (strcasecmp (st->name, string) == 0) {
	    *salttype = st->type;
	    return 0;
	}
    }
    krb5_set_error_message(context, HEIM_ERR_SALTTYPE_NOSUPP,
			   N_("salttype %s not supported", ""), string);
    return HEIM_ERR_SALTTYPE_NOSUPP;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_get_pw_salt(krb5_context context,
		 krb5_const_principal principal,
		 krb5_salt *salt)
{
    size_t len;
    size_t i;
    krb5_error_code ret;
    char *p;

    salt->salttype = KRB5_PW_SALT;
    len = strlen(principal->realm);
    for (i = 0; i < principal->name.name_string.len; ++i)
	len += strlen(principal->name.name_string.val[i]);
    ret = krb5_data_alloc (&salt->saltvalue, len);
    if (ret)
	return ret;
    p = salt->saltvalue.data;
    memcpy (p, principal->realm, strlen(principal->realm));
    p += strlen(principal->realm);
    for (i = 0; i < principal->name.name_string.len; ++i) {
	memcpy (p,
		principal->name.name_string.val[i],
		strlen(principal->name.name_string.val[i]));
	p += strlen(principal->name.name_string.val[i]);
    }
    return 0;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_free_salt(krb5_context context,
	       krb5_salt salt)
{
    krb5_data_free(&salt.saltvalue);
    return 0;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_string_to_key_data (krb5_context context,
			 krb5_enctype enctype,
			 krb5_data password,
			 krb5_principal principal,
			 krb5_keyblock *key)
{
    krb5_error_code ret;
    krb5_salt salt;

    ret = krb5_get_pw_salt(context, principal, &salt);
    if(ret)
	return ret;
    ret = krb5_string_to_key_data_salt(context, enctype, password, salt, key);
    krb5_free_salt(context, salt);
    return ret;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_string_to_key (krb5_context context,
		    krb5_enctype enctype,
		    const char *password,
		    krb5_principal principal,
		    krb5_keyblock *key)
{
    krb5_data pw;
    pw.data = rk_UNCONST(password);
    pw.length = strlen(password);
    return krb5_string_to_key_data(context, enctype, pw, principal, key);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_string_to_key_data_salt (krb5_context context,
			      krb5_enctype enctype,
			      krb5_data password,
			      krb5_salt salt,
			      krb5_keyblock *key)
{
    krb5_data opaque;
    krb5_data_zero(&opaque);
    return krb5_string_to_key_data_salt_opaque(context, enctype, password,
					       salt, opaque, key);
}

/*
 * Do a string -> key for encryption type `enctype' operation on
 * `password' (with salt `salt' and the enctype specific data string
 * `opaque'), returning the resulting key in `key'
 */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_string_to_key_data_salt_opaque (krb5_context context,
				     krb5_enctype enctype,
				     krb5_data password,
				     krb5_salt salt,
				     krb5_data opaque,
				     krb5_keyblock *key)
{
    struct _krb5_encryption_type *et =_krb5_find_enctype(enctype);
    struct salt_type *st;
    if(et == NULL) {
	krb5_set_error_message(context, KRB5_PROG_ETYPE_NOSUPP,
			       N_("encryption type %d not supported", ""),
			       enctype);
	return KRB5_PROG_ETYPE_NOSUPP;
    }
    for(st = et->keytype->string_to_key; st && st->type; st++)
	if(st->type == salt.salttype)
	    return (*st->string_to_key)(context, enctype, password,
					salt, opaque, key);
    krb5_set_error_message(context, HEIM_ERR_SALTTYPE_NOSUPP,
			   N_("salt type %u not supported", ""),
			   salt.salttype);
    return HEIM_ERR_SALTTYPE_NOSUPP;
}

/*
 * Do a string -> key for encryption type `enctype' operation on the
 * string `password' (with salt `salt'), returning the resulting key
 * in `key'
 */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_string_to_key_salt (krb5_context context,
			 krb5_enctype enctype,
			 const char *password,
			 krb5_salt salt,
			 krb5_keyblock *key)
{
    krb5_data pw;
    pw.data = rk_UNCONST(password);
    pw.length = strlen(password);
    return krb5_string_to_key_data_salt(context, enctype, pw, salt, key);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_string_to_key_salt_opaque (krb5_context context,
				krb5_enctype enctype,
				const char *password,
				krb5_salt salt,
				krb5_data opaque,
				krb5_keyblock *key)
{
    krb5_data pw;
    pw.data = rk_UNCONST(password);
    pw.length = strlen(password);
    return krb5_string_to_key_data_salt_opaque(context, enctype,
					       pw, salt, opaque, key);
}


KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_string_to_key_derived(krb5_context context,
			   const void *str,
			   size_t len,
			   krb5_enctype etype,
			   krb5_keyblock *key)
{
#ifdef HEIM_KRB5_DES3
    struct _krb5_encryption_type *et = _krb5_find_enctype(etype);
    krb5_error_code ret;
    struct _krb5_key_data kd;
    size_t keylen;
    u_char *tmp;

    if(et == NULL) {
	krb5_set_error_message (context, KRB5_PROG_ETYPE_NOSUPP,
				N_("encryption type %d not supported", ""),
				etype);
	return KRB5_PROG_ETYPE_NOSUPP;
    }
    keylen = et->keytype->bits / 8;

    ALLOC(kd.key, 1);
    if(kd.key == NULL) {
	krb5_set_error_message (context, ENOMEM,
				N_("malloc: out of memory", ""));
	return ENOMEM;
    }
    ret = krb5_data_alloc(&kd.key->keyvalue, et->keytype->size);
    if(ret) {
	free(kd.key);
	return ret;
    }
    kd.key->keytype = etype;
    tmp = malloc (keylen);
    if(tmp == NULL) {
	krb5_free_keyblock(context, kd.key);
	krb5_set_error_message (context, ENOMEM, N_("malloc: out of memory", ""));
	return ENOMEM;
    }
    ret = _krb5_n_fold(str, len, tmp, keylen);
    if (ret) {
	free(tmp);
	krb5_set_error_message (context, ENOMEM, N_("malloc: out of memory", ""));
	return ret;
    }
    kd.schedule = NULL;
    _krb5_DES3_random_to_key(context, kd.key, tmp, keylen);
    memset(tmp, 0, keylen);
    free(tmp);
    ret = _krb5_derive_key(context,
			   et,
			   &kd,
			   "kerberos", /* XXX well known constant */
			   strlen("kerberos"));
    if (ret) {
	_krb5_free_key_data(context, &kd, et);
	return ret;
    }
    ret = krb5_copy_keyblock_contents(context, kd.key, key);
    _krb5_free_key_data(context, &kd, et);
    return ret;
#else
    krb5_set_error_message (context, KRB5_PROG_ETYPE_NOSUPP,
			    N_("encryption type %d not supported", ""),
			    etype);
    return KRB5_PROG_ETYPE_NOSUPP;
#endif
}
