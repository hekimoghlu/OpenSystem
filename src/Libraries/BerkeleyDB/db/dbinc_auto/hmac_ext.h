/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 13, 2022.
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
#ifndef	_hmac_ext_h_
#define	_hmac_ext_h_

#if defined(__cplusplus)
extern "C" {
#endif

void __db_chksum __P((void *, u_int8_t *, size_t, u_int8_t *, u_int8_t *));
void __db_derive_mac __P((u_int8_t *, size_t, u_int8_t *));
int __db_check_chksum __P((ENV *, void *, DB_CIPHER *, u_int8_t *, void *, size_t, int));
void __db_SHA1Transform __P((u_int32_t *, unsigned char *));
void __db_SHA1Init __P((SHA1_CTX *));
void __db_SHA1Update __P((SHA1_CTX *, unsigned char *, size_t));
void __db_SHA1Final __P((unsigned char *, SHA1_CTX *));

#if defined(__cplusplus)
}
#endif
#endif /* !_hmac_ext_h_ */
