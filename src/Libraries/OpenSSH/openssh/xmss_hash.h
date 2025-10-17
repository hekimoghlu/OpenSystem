/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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

#ifdef WITH_XMSS
/* $OpenBSD: xmss_hash.h,v 1.2 2018/02/26 03:56:44 dtucker Exp $ */
/*
hash.h version 20160722
Andreas HÃ¼lsing
Joost Rijneveld
Public domain.
*/

#ifndef HASH_H
#define HASH_H

#define IS_LITTLE_ENDIAN 1

unsigned char* addr_to_byte(unsigned char *bytes, const uint32_t addr[8]);
int prf(unsigned char *out, const unsigned char *in, const unsigned char *key, unsigned int keylen);
int h_msg(unsigned char *out,const unsigned char *in,unsigned long long inlen, const unsigned char *key, const unsigned int keylen, const unsigned int n);
int hash_h(unsigned char *out, const unsigned char *in, const unsigned char *pub_seed, uint32_t addr[8], const unsigned int n);
int hash_f(unsigned char *out, const unsigned char *in, const unsigned char *pub_seed, uint32_t addr[8], const unsigned int n);

#endif
#endif /* WITH_XMSS */
