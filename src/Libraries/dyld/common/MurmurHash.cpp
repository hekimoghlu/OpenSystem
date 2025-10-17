/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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
 Portions derived from:
 
 ------------------------------------------------------------------------------
 // MurmurHash2 was written by Austin Appleby, and is placed in the public
 // domain. The author hereby disclaims copyright to this source code.
 Source is https://github.com/aappleby/smhasher/blob/master/src/MurmurHash2.cpp
------------------------------------------------------------------------------
*/

#include "MurmurHash.h"

uint64_t murmurHash(const void* key, int len, uint64_t seed)
{
  const uint64_t magic = 0xc6a4a7935bd1e995ULL;
  const int salt = 47;

  uint64_t hash = seed ^ (len * magic);

  const uint64_t * data = (const uint64_t *)key;
  const uint64_t * end = data + (len/8);

  while(data != end)
  {
    uint64_t val = *data++;

    val *= magic;
    val ^= val >> salt;
    val *= magic;

    hash ^= val;
    hash *= magic;
  }

  const unsigned char * data2 = (const unsigned char*)data;

  switch(len & 7)
  {
      case 7: hash ^= uint64_t(data2[6]) << 48; [[fallthrough]];
      case 6: hash ^= uint64_t(data2[5]) << 40; [[fallthrough]];
      case 5: hash ^= uint64_t(data2[4]) << 32; [[fallthrough]];
      case 4: hash ^= uint64_t(data2[3]) << 24; [[fallthrough]];
      case 3: hash ^= uint64_t(data2[2]) << 16; [[fallthrough]];
      case 2: hash ^= uint64_t(data2[1]) << 8; [[fallthrough]];
      case 1: hash ^= uint64_t(data2[0]);
  };
    hash *= magic;

    hash ^= hash >> salt;
    hash *= magic;
    hash ^= hash >> salt;

  return hash;
}
