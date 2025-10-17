/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 5, 2022.
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

#ifndef HEIMDAL_SMALLER
#define DES3_OLD_ENCTYPE 1
#endif

struct _krb5_checksum_type *_krb5_checksum_types[] = {
    &_krb5_checksum_none,
#ifdef HEIM_KRB5_DES
    &_krb5_checksum_crc32,
    &_krb5_checksum_rsa_md4,
    &_krb5_checksum_rsa_md4_des,
    &_krb5_checksum_rsa_md5_des,
#endif
#ifdef DES3_OLD_ENCTYPE
    &_krb5_checksum_rsa_md5_des3,
#endif
    &_krb5_checksum_rsa_md5,
    &_krb5_checksum_sha1,
    &_krb5_checksum_hmac_sha1_des3,
    &_krb5_checksum_hmac_sha1_aes128,
    &_krb5_checksum_hmac_sha1_aes256,
    &_krb5_checksum_hmac_md5
};

int _krb5_num_checksums
	= sizeof(_krb5_checksum_types) / sizeof(_krb5_checksum_types[0]);

/*
 * these should currently be in reverse preference order.
 * (only relevant for !F_PSEUDO) */

struct _krb5_encryption_type *_krb5_etypes[] = {
    &_krb5_enctype_aes256_cts_hmac_sha1,
    &_krb5_enctype_aes128_cts_hmac_sha1,
#ifdef HEIM_KRB5_DES3
    &_krb5_enctype_des3_cbc_sha1,
    &_krb5_enctype_des3_cbc_none, /* used by the gss-api mech */
#ifdef DES3_OLD_ENCTYPE
    &_krb5_enctype_des3_cbc_md5,
    &_krb5_enctype_old_des3_cbc_sha1,
#endif
#endif
#ifdef HEIM_KRB5_ARCFOUR
    &_krb5_enctype_arcfour_hmac_md5,
#endif
#ifdef HEIM_KRB5_DES
    &_krb5_enctype_des_cbc_md5,
    &_krb5_enctype_des_cbc_md4,
    &_krb5_enctype_des_cbc_crc,
    &_krb5_enctype_des_cbc_none,
#ifndef __APPLE_PRIVATE__
    &_krb5_enctype_des_cfb64_none,
    &_krb5_enctype_des_pcbc_none,
#endif
#endif
    &_krb5_enctype_null
};

int _krb5_num_etypes = sizeof(_krb5_etypes) / sizeof(_krb5_etypes[0]);

struct _krb5_etypes_deprected _krb5_deprecated_etypes[] = {
#ifndef HEIM_KRB5_DES
    {
	ETYPE_DES_CBC_MD5,
	"des-cbc-md5-deprecated",
    },
    {
	ETYPE_DES_CBC_MD4,
	"des-cbc-md4-deprecated",
    },
    {
	ETYPE_DES_CBC_CRC,
	"des-cbc-crc-deprecated",
    }
#endif
};

int _krb5_num_deprecated_etypes = sizeof(_krb5_deprecated_etypes) / sizeof(_krb5_deprecated_etypes[0]);
