/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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
#include <CommonCrypto/CommonCryptor.h>
#ifndef __APPLE_TARGET_EMBEDDED__
#include <CommonCrypto/CommonCryptor.h>
#include <CommonCrypto/CommonCryptorSPI.h>
#endif

#ifdef HEIM_KRB5_DES

#ifdef ENABLE_AFS_STRING_TO_KEY

/* This defines the Andrew string_to_key function.  It accepts a password
 * string as input and converts it via a one-way encryption algorithm to a DES
 * encryption key.  It is compatible with the original Andrew authentication
 * service password database.
 */

/*
 * Short passwords, i.e 8 characters or less.
 */
static void
krb5_DES_AFS3_CMU_string_to_key (krb5_data pw,
				 krb5_data cell,
				 DES_cblock *key)
{
    char  password[8+1];	/* crypt is limited to 8 chars anyway */
    size_t   i;

    for(i = 0; i < 8; i++) {
	char c = ((i < pw.length) ? ((char*)pw.data)[i] : 0) ^
	    ((i < cell.length) ?
	     tolower(((unsigned char*)cell.data)[i]) : 0);
	password[i] = c ? c : 'X';
    }
    password[8] = '\0';

    memcpy(key, crypt(password, "p1") + 2, sizeof(DES_cblock));

    /* parity is inserted into the LSB so left shift each byte up one
       bit. This allows ascii characters with a zero MSB to retain as
       much significance as possible. */
    for (i = 0; i < sizeof(DES_cblock); i++)
	((unsigned char*)key)[i] <<= 1;
    CCDesSetOddParity(key, sizeof(*key));
}

/*
 * Long passwords, i.e 9 characters or more.
 */
static void
krb5_DES_AFS3_Transarc_string_to_key (krb5_data pw,
				      krb5_data cell,
				      DES_cblock *key)
{
    DES_key_schedule schedule;
    DES_cblock temp_key;
    DES_cblock ivec;
    char password[512];
    size_t passlen;

    memcpy(password, pw.data, min(pw.length, sizeof(password)));
    if(pw.length < sizeof(password)) {
	size_t len = min(cell.length, sizeof(password) - pw.length);
	size_t i;

	memcpy(password + pw.length, cell.data, len);
	for (i = pw.length; i < pw.length + len; ++i)
	    password[i] = tolower((unsigned char)password[i]);
    }
    passlen = min(sizeof(password), pw.length + cell.length);
    memcpy(&ivec, "kerberos", 8);
    memcpy(&temp_key, "kerberos", 8);
    CCDesCBCCksum(password, &ivec, passlen, temp_key, sizeof(temp_key), &ivec);

    memcpy(&temp_key, &ivec, 8);
    CCDesCBCCksum(password, &key, passlen, temp_key, sizeof(temp_key), &ivec);
    memset(&schedule, 0, sizeof(schedule));
    memset(&temp_key, 0, sizeof(temp_key));
    memset(&ivec, 0, sizeof(ivec));
    memset(password, 0, sizeof(password));

    CCDesSetOddParity(key, sizeof(*key));
}

static krb5_error_code
DES_AFS3_string_to_key(krb5_context context,
		       krb5_enctype enctype,
		       krb5_data password,
		       krb5_salt salt,
		       krb5_data opaque,
		       krb5_keyblock *key)
{
    DES_cblock tmp;
    if(password.length > 8)
	krb5_DES_AFS3_Transarc_string_to_key(password, salt.saltvalue, &tmp);
    else
	krb5_DES_AFS3_CMU_string_to_key(password, salt.saltvalue, &tmp);
    key->keytype = enctype;
    krb5_data_copy(&key->keyvalue, tmp, sizeof(tmp));
    memset(&key, 0, sizeof(key));
    return 0;
}
#endif /* ENABLE_AFS_STRING_TO_KEY */

static void
DES_string_to_key_int(unsigned char *data, size_t length, DES_cblock *key)
{
    DES_key_schedule schedule;
    size_t i;
    int reverse = 0;
    unsigned char *p;

    unsigned char swap[] = { 0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe,
			     0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf };
    memset(key, 0, 8);

    p = (unsigned char*)key;
    for (i = 0; i < length; i++) {
	unsigned char tmp = data[i];
	if (!reverse)
	    *p++ ^= (tmp << 1);
	else
	    *--p ^= (swap[tmp & 0xf] << 4) | swap[(tmp & 0xf0) >> 4];
	if((i % 8) == 7)
	    reverse = !reverse;
    }
    CCDesSetOddParity(key, sizeof(*key));
    if(CCDesIsWeakKey(key, sizeof(*key)))
	(*key)[7] ^= 0xF0;
#ifndef __APPLE_PRIVATE__
    DES_set_key_unchecked(key, &schedule);
    DES_cbc_cksum((void*)data, key, length, &schedule, key);
#else
    CCDesCBCCksum(data, key, length, key, sizeof(*key), key);
#endif
    memset(&schedule, 0, sizeof(schedule));
    CCDesSetOddParity(key, sizeof(*key));
    if(CCDesIsWeakKey(key, sizeof(*key)))
	(*key)[7] ^= 0xF0;
}

static krb5_error_code
krb5_DES_string_to_key(krb5_context context,
		       krb5_enctype enctype,
		       krb5_data password,
		       krb5_salt salt,
		       krb5_data opaque,
		       krb5_keyblock *key)
{
    unsigned char *s;
    size_t len;
    DES_cblock tmp;

#ifdef ENABLE_AFS_STRING_TO_KEY
    if (opaque.length == 1) {
	unsigned long v;
	_krb5_get_int(opaque.data, &v, 1);
	if (v == 1)
	    return DES_AFS3_string_to_key(context, enctype, password,
					  salt, opaque, key);
    }
#endif

    len = password.length + salt.saltvalue.length;
    s = malloc(len);
    if(len > 0 && s == NULL) {
	krb5_set_error_message(context, ENOMEM, N_("malloc: out of memory", ""));
	return ENOMEM;
    }
    memcpy(s, password.data, password.length);
    memcpy(s + password.length, salt.saltvalue.data, salt.saltvalue.length);
    DES_string_to_key_int(s, len, &tmp);
    key->keytype = enctype;
    krb5_data_copy(&key->keyvalue, tmp, sizeof(tmp));
    memset(&tmp, 0, sizeof(tmp));
    memset(s, 0, len);
    free(s);
    return 0;
}

struct salt_type _krb5_des_salt[] = {
    {
	KRB5_PW_SALT,
	"pw-salt",
	krb5_DES_string_to_key
    },
#ifdef ENABLE_AFS_STRING_TO_KEY
    {
	KRB5_AFS3_SALT,
	"afs3-salt",
	DES_AFS3_string_to_key
    },
#endif
    { 0 }
};

#endif /* HEIM_KRB5_DES */
