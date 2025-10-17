/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 21, 2022.
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
/*
SHA-1 in C
By Steve Reid <steve@edmweb.com>
100% Public Domain
*/
/* Header portion split from main code for convenience (AYB 3/02/98) */

#ifndef __SHA1_H__

#define __SHA1_H__

#include <sys/types.h>

/*
Test Vectors (from FIPS PUB 180-1)
"abc"
  A9993E36 4706816A BA3E2571 7850C26C 9CD0D89D
"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
  84983E44 1C3BD26E BAAE4AA1 F95129E5 E54670F1
A million repetitions of "a"
  34AA973C D4C4DAA4 F61EEB2B DBAD2731 6534016F
*/

/* Apple change - define this in the source file which uses it */
/* #define LITTLE_ENDIAN  This should be #define'd if true. */
#define SHA1HANDSOFF /* Copies data before messing with it. */

//Context declaration
typedef struct {
    u_int32_t state[5];
    u_int32_t count[2];
    unsigned char buffer[64];
} YSHA1_CTX;

//Function forward declerations
__private_extern__ void YSHA1Transform(u_int32_t state[5],
    const unsigned char buffer[64]);
__private_extern__ void YSHA1Init(YSHA1_CTX* context);
__private_extern__ void YSHA1Update(YSHA1_CTX* context,
    const unsigned char* data, unsigned int len);
__private_extern__ void YSHA1Final(unsigned char digest[20],
    YSHA1_CTX* context);

#endif	/* __SHA1_H__ */
