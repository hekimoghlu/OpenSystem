/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 18, 2022.
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
#ifndef HEADER_SEED_LOCL_H
#define HEADER_SEED_LOCL_H

#include "openssl/e_os2.h"
#include <openssl/seed.h>


#ifdef SEED_LONG /* need 32-bit type */
typedef unsigned long seed_word;
#else
typedef unsigned int seed_word;
#endif


#ifdef  __cplusplus
extern "C" {
#endif

#define G_FUNC(v)       \
        SS[0][(unsigned char)      (v) & 0xff] ^ SS[1][(unsigned char) ((v)>>8) & 0xff] ^ \
        SS[2][(unsigned char)((v)>>16) & 0xff] ^ SS[3][(unsigned char)((v)>>24) & 0xff]

#define char2word(c, i)  \
        (i) = ((((seed_word)(c)[0]) << 24) | (((seed_word)(c)[1]) << 16) | (((seed_word)(c)[2]) << 8) | ((seed_word)(c)[3]))

#define word2char(l, c)  \
        *((c)+0) = (unsigned char)((l)>>24) & 0xff; \
        *((c)+1) = (unsigned char)((l)>>16) & 0xff; \
        *((c)+2) = (unsigned char)((l)>> 8) & 0xff; \
        *((c)+3) = (unsigned char)((l))     & 0xff

#define KEYSCHEDULE_UPDATE0(T0, T1, X1, X2, X3, X4, KC)  \
        (T0) = (X3);                                     \
        (X3) = (((X3)<<8) ^ ((X4)>>24)) & 0xffffffff;    \
        (X4) = (((X4)<<8) ^ ((T0)>>24)) & 0xffffffff;    \
        (T0) = ((X1) + (X3) - (KC))     & 0xffffffff;    \
        (T1) = ((X2) + (KC) - (X4))     & 0xffffffff

#define KEYSCHEDULE_UPDATE1(T0, T1, X1, X2, X3, X4, KC)  \
        (T0) = (X1);                                     \
        (X1) = (((X1)>>8) ^ ((X2)<<24)) & 0xffffffff;    \
        (X2) = (((X2)>>8) ^ ((T0)<<24)) & 0xffffffff;    \
        (T0) = ((X1) + (X3) - (KC))     & 0xffffffff;     \
        (T1) = ((X2) + (KC) - (X4))     & 0xffffffff

#define KEYUPDATE_TEMP(T0, T1, K)   \
        (K)[0] = G_FUNC((T0));      \
        (K)[1] = G_FUNC((T1))

#define XOR_SEEDBLOCK(DST, SRC)      \
        ((DST))[0] ^= ((SRC))[0];    \
        ((DST))[1] ^= ((SRC))[1];    \
        ((DST))[2] ^= ((SRC))[2];    \
        ((DST))[3] ^= ((SRC))[3]

#define MOV_SEEDBLOCK(DST, SRC)      \
        ((DST))[0] = ((SRC))[0];     \
        ((DST))[1] = ((SRC))[1];     \
        ((DST))[2] = ((SRC))[2];     \
        ((DST))[3] = ((SRC))[3]

# define CHAR2WORD(C, I)              \
        char2word((C),    (I)[0]);    \
        char2word((C+4),  (I)[1]);    \
        char2word((C+8),  (I)[2]);    \
        char2word((C+12), (I)[3])

# define WORD2CHAR(I, C)              \
        word2char((I)[0], (C));       \
        word2char((I)[1], (C+4));     \
        word2char((I)[2], (C+8));     \
        word2char((I)[3], (C+12))

# define E_SEED(T0, T1, X1, X2, X3, X4, rbase)   \
        (T0) = (X3) ^ (ks->data)[(rbase)];       \
        (T1) = (X4) ^ (ks->data)[(rbase)+1];     \
        (T1) ^= (T0);                            \
        (T1) = G_FUNC((T1));                     \
        (T0) = ((T0) + (T1)) & 0xffffffff;       \
        (T0) = G_FUNC((T0));                     \
        (T1) = ((T1) + (T0)) & 0xffffffff;       \
        (T1) = G_FUNC((T1));                     \
        (T0) = ((T0) + (T1)) & 0xffffffff;       \
        (X1) ^= (T0);                            \
        (X2) ^= (T1)

#ifdef  __cplusplus
}
#endif

#endif /* HEADER_SEED_LOCL_H */
