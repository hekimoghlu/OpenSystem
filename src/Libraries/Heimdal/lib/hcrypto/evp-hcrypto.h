/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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
/* $Id$ */

#ifndef HEIM_EVP_HCRYPTO_H
#define HEIM_EVP_HCRYPTO_H 1

/* symbol renaming */
#define EVP_hcrypto_md4 hc_EVP_hcrypto_md4
#define EVP_hcrypto_md5 hc_EVP_hcrypto_md5
#define EVP_hcrypto_sha1 hc_EVP_hcrypto_sha1
#define EVP_hcrypto_sha256 hc_EVP_hcrypto_sha256
#define EVP_hcrypto_sha384 hc_EVP_hcrypto_sha384
#define EVP_hcrypto_sha512 hc_EVP_hcrypto_sha512
#define EVP_hcrypto_des_cbc hc_EVP_hcrypto_des_cbc
#define EVP_hcrypto_des_ede3_cbc hc_EVP_hcrypto_des_ede3_cbc
#define EVP_hcrypto_aes_128_cbc hc_EVP_hcrypto_aes_128_cbc
#define EVP_hcrypto_aes_192_cbc hc_EVP_hcrypto_aes_192_cbc
#define EVP_hcrypto_aes_256_cbc hc_EVP_hcrypto_aes_256_cbc
#define EVP_hcrypto_aes_128_cfb8 hc_EVP_hcrypto_aes_128_cfb8
#define EVP_hcrypto_aes_192_cfb8 hc_EVP_hcrypto_aes_192_cfb8
#define EVP_hcrypto_aes_256_cfb8 hc_EVP_hcrypto_aes_256_cfb8
#define EVP_hcrypto_rc4 hc_EVP_hcrypto_rc4
#define EVP_hcrypto_rc4_40 hc_EVP_hcrypto_rc4_40
#define EVP_hcrypto_rc2_40_cbc hc_EVP_hcrypto_rc2_40_cbc
#define EVP_hcrypto_rc2_64_cbc hc_EVP_hcrypto_rc2_64_cbc
#define EVP_hcrypto_rc2_cbc hc_EVP_hcrypto_rc2_cbc
#define EVP_hcrypto_camellia_128_cbc hc_EVP_hcrypto_camellia_128_cbc
#define EVP_hcrypto_camellia_192_cbc hc_EVP_hcrypto_camellia_192_cbc
#define EVP_hcrypto_camellia_256_cbc hc_EVP_hcrypto_camellia_256_cbc

/*
 *
 */

HC_CPP_BEGIN

const EVP_MD * EVP_hcrypto_md4(void);
const EVP_MD * EVP_hcrypto_md5(void);
const EVP_MD * EVP_hcrypto_sha1(void);
const EVP_MD * EVP_hcrypto_sha256(void);
const EVP_MD * EVP_hcrypto_sha384(void);
const EVP_MD * EVP_hcrypto_sha512(void);

const EVP_CIPHER * EVP_hcrypto_rc4(void);
const EVP_CIPHER * EVP_hcrypto_rc4_40(void);

const EVP_CIPHER * EVP_hcrypto_rc2_cbc(void);
const EVP_CIPHER * EVP_hcrypto_rc2_40_cbc(void);
const EVP_CIPHER * EVP_hcrypto_rc2_64_cbc(void);

const EVP_CIPHER * EVP_hcrypto_des_cbc(void);
const EVP_CIPHER * EVP_hcrypto_des_ede3_cbc(void);

const EVP_CIPHER * EVP_hcrypto_aes_128_cbc(void);
const EVP_CIPHER * EVP_hcrypto_aes_192_cbc(void);
const EVP_CIPHER * EVP_hcrypto_aes_256_cbc(void);

const EVP_CIPHER * EVP_hcrypto_aes_128_cfb8(void);
const EVP_CIPHER * EVP_hcrypto_aes_192_cfb8(void);
const EVP_CIPHER * EVP_hcrypto_aes_256_cfb8(void);

const EVP_CIPHER * EVP_hcrypto_camellia_128_cbc(void);
const EVP_CIPHER * EVP_hcrypto_camellia_192_cbc(void);
const EVP_CIPHER * EVP_hcrypto_camellia_256_cbc(void);


HC_CPP_END

#endif /* HEIM_EVP_HCRYPTO_H */
