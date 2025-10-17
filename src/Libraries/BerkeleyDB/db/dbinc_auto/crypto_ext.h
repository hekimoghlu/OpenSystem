/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
#ifndef	_crypto_ext_h_
#define	_crypto_ext_h_

#if defined(__cplusplus)
extern "C" {
#endif

int __aes_setup __P((ENV *, DB_CIPHER *));
u_int __aes_adj_size __P((size_t));
int __aes_close __P((ENV *, void *));
int __aes_decrypt __P((ENV *, void *, void *, u_int8_t *, size_t));
int __aes_encrypt __P((ENV *, void *, void *, u_int8_t *, size_t));
int __aes_init __P((ENV *, DB_CIPHER *));
int __crypto_env_close __P((ENV *));
int __crypto_env_refresh __P((ENV *));
int __crypto_algsetup __P((ENV *, DB_CIPHER *, u_int32_t, int));
int __crypto_decrypt_meta __P((ENV *, DB *, u_int8_t *, int));
int __crypto_set_passwd __P((ENV *, ENV *));
int __db_generate_iv __P((ENV *, u_int32_t *));
int __db_rijndaelKeySetupEnc __P((u32 *, const u8 *, int));
int __db_rijndaelKeySetupDec __P((u32 *, const u8 *, int));
void __db_rijndaelEncrypt __P((u32 *, int, const u8 *, u8 *));
void __db_rijndaelDecrypt __P((u32 *, int, const u8 *, u8 *));
void __db_rijndaelEncryptRound __P((const u32 *, int, u8 *, int));
void __db_rijndaelDecryptRound __P((const u32 *, int, u8 *, int));
int __db_makeKey __P((keyInstance *, int, int, char *));
int __db_cipherInit __P((cipherInstance *, int, char *));
int __db_blockEncrypt __P((cipherInstance *, keyInstance *, u_int8_t *, size_t, u_int8_t *));
int __db_padEncrypt __P((cipherInstance *, keyInstance *, u_int8_t *, int, u_int8_t *));
int __db_blockDecrypt __P((cipherInstance *, keyInstance *, u_int8_t *, size_t, u_int8_t *));
int __db_padDecrypt __P((cipherInstance *, keyInstance *, u_int8_t *, int, u_int8_t *));
int __db_cipherUpdateRounds __P((cipherInstance *, keyInstance *, u_int8_t *, int, u_int8_t *, int));

#if defined(__cplusplus)
}
#endif
#endif /* !_crypto_ext_h_ */
