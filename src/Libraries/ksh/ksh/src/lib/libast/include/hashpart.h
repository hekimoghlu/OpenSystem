/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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
#pragma prototyped
/*
 * K. P. Vo
 * G. S. Fowler
 * AT&T Research
 *
 * ``the best'' combined linear congruent checksum/hash/PRNG
 */

#ifndef _HASHPART_H
#define _HASHPART_H

#define HASH_ADD(h)	(0x9c39c33dL)

#if __sparc__ || __sparc || sparc

#define HASH_A(h,n)	((((h) << 2) - (h)) << (n))
#define HASH_B(h,n)	((((h) << 4) - (h)) << (n))
#define HASH_C(h,n)	((HASH_A(h,7) + HASH_B(h,0)) << (n))
#define HASH_MPY(h)	(HASH_C(h,22)+HASH_C(h,10)+HASH_A(h,6)+HASH_A(h,3)+(h))

#else

#define HASH_MPY(h)	((h)*0x63c63cd9L)

#endif

#define HASHPART(h,c)	(h = HASH_MPY(h) + HASH_ADD(h) + (c))

#endif
