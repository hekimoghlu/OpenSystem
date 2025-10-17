/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 8, 2024.
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
#include <IndexStoreDB_LLVMSupport/toolchain_Support_MD5.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_ArrayRef.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_StringRef.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_Endian.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_Format.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_raw_ostream.h>
#include <array>
#include <cstdint>
#include <cstring>

// The basic MD5 functions.

// F and G are optimized compared to their RFC 1321 definitions for
// architectures that lack an AND-NOT instruction, just like in Colin Plumb's
// implementation.
#define F(x, y, z) ((z) ^ ((x) & ((y) ^ (z))))
#define G(x, y, z) ((y) ^ ((z) & ((x) ^ (y))))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | ~(z)))

// The MD5 transformation for all four rounds.
#define STEP(f, a, b, c, d, x, t, s)                                           \
  (a) += f((b), (c), (d)) + (x) + (t);                                         \
  (a) = (((a) << (s)) | (((a) & 0xffffffff) >> (32 - (s))));                   \
  (a) += (b);

// SET reads 4 input bytes in little-endian byte order and stores them
// in a properly aligned word in host byte order.
#define SET(n)                                                                 \
  (block[(n)] =                                                                \
       (MD5_u32plus) ptr[(n) * 4] | ((MD5_u32plus) ptr[(n) * 4 + 1] << 8) |    \
       ((MD5_u32plus) ptr[(n) * 4 + 2] << 16) |                                \
       ((MD5_u32plus) ptr[(n) * 4 + 3] << 24))
#define GET(n) (block[(n)])

using namespace toolchain;

/// This processes one or more 64-byte data blocks, but does NOT update
///the bit counters.  There are no alignment requirements.
const uint8_t *MD5::body(ArrayRef<uint8_t> Data) {
  const uint8_t *ptr;
  MD5_u32plus a, b, c, d;
  MD5_u32plus saved_a, saved_b, saved_c, saved_d;
  unsigned long Size = Data.size();

  ptr = Data.data();

  a = this->a;
  b = this->b;
  c = this->c;
  d = this->d;

  do {
    saved_a = a;
    saved_b = b;
    saved_c = c;
    saved_d = d;

    // Round 1
    STEP(F, a, b, c, d, SET(0), 0xd76aa478, 7)
    STEP(F, d, a, b, c, SET(1), 0xe8c7b756, 12)
    STEP(F, c, d, a, b, SET(2), 0x242070db, 17)
    STEP(F, b, c, d, a, SET(3), 0xc1bdceee, 22)
    STEP(F, a, b, c, d, SET(4), 0xf57c0faf, 7)
    STEP(F, d, a, b, c, SET(5), 0x4787c62a, 12)
    STEP(F, c, d, a, b, SET(6), 0xa8304613, 17)
    STEP(F, b, c, d, a, SET(7), 0xfd469501, 22)
    STEP(F, a, b, c, d, SET(8), 0x698098d8, 7)
    STEP(F, d, a, b, c, SET(9), 0x8b44f7af, 12)
    STEP(F, c, d, a, b, SET(10), 0xffff5bb1, 17)
    STEP(F, b, c, d, a, SET(11), 0x895cd7be, 22)
    STEP(F, a, b, c, d, SET(12), 0x6b901122, 7)
    STEP(F, d, a, b, c, SET(13), 0xfd987193, 12)
    STEP(F, c, d, a, b, SET(14), 0xa679438e, 17)
    STEP(F, b, c, d, a, SET(15), 0x49b40821, 22)

    // Round 2
    STEP(G, a, b, c, d, GET(1), 0xf61e2562, 5)
    STEP(G, d, a, b, c, GET(6), 0xc040b340, 9)
    STEP(G, c, d, a, b, GET(11), 0x265e5a51, 14)
    STEP(G, b, c, d, a, GET(0), 0xe9b6c7aa, 20)
    STEP(G, a, b, c, d, GET(5), 0xd62f105d, 5)
    STEP(G, d, a, b, c, GET(10), 0x02441453, 9)
    STEP(G, c, d, a, b, GET(15), 0xd8a1e681, 14)
    STEP(G, b, c, d, a, GET(4), 0xe7d3fbc8, 20)
    STEP(G, a, b, c, d, GET(9), 0x21e1cde6, 5)
    STEP(G, d, a, b, c, GET(14), 0xc33707d6, 9)
    STEP(G, c, d, a, b, GET(3), 0xf4d50d87, 14)
    STEP(G, b, c, d, a, GET(8), 0x455a14ed, 20)
    STEP(G, a, b, c, d, GET(13), 0xa9e3e905, 5)
    STEP(G, d, a, b, c, GET(2), 0xfcefa3f8, 9)
    STEP(G, c, d, a, b, GET(7), 0x676f02d9, 14)
    STEP(G, b, c, d, a, GET(12), 0x8d2a4c8a, 20)

    // Round 3
    STEP(H, a, b, c, d, GET(5), 0xfffa3942, 4)
    STEP(H, d, a, b, c, GET(8), 0x8771f681, 11)
    STEP(H, c, d, a, b, GET(11), 0x6d9d6122, 16)
    STEP(H, b, c, d, a, GET(14), 0xfde5380c, 23)
    STEP(H, a, b, c, d, GET(1), 0xa4beea44, 4)
    STEP(H, d, a, b, c, GET(4), 0x4bdecfa9, 11)
    STEP(H, c, d, a, b, GET(7), 0xf6bb4b60, 16)
    STEP(H, b, c, d, a, GET(10), 0xbebfbc70, 23)
    STEP(H, a, b, c, d, GET(13), 0x289b7ec6, 4)
    STEP(H, d, a, b, c, GET(0), 0xeaa127fa, 11)
    STEP(H, c, d, a, b, GET(3), 0xd4ef3085, 16)
    STEP(H, b, c, d, a, GET(6), 0x04881d05, 23)
    STEP(H, a, b, c, d, GET(9), 0xd9d4d039, 4)
    STEP(H, d, a, b, c, GET(12), 0xe6db99e5, 11)
    STEP(H, c, d, a, b, GET(15), 0x1fa27cf8, 16)
    STEP(H, b, c, d, a, GET(2), 0xc4ac5665, 23)

    // Round 4
    STEP(I, a, b, c, d, GET(0), 0xf4292244, 6)
    STEP(I, d, a, b, c, GET(7), 0x432aff97, 10)
    STEP(I, c, d, a, b, GET(14), 0xab9423a7, 15)
    STEP(I, b, c, d, a, GET(5), 0xfc93a039, 21)
    STEP(I, a, b, c, d, GET(12), 0x655b59c3, 6)
    STEP(I, d, a, b, c, GET(3), 0x8f0ccc92, 10)
    STEP(I, c, d, a, b, GET(10), 0xffeff47d, 15)
    STEP(I, b, c, d, a, GET(1), 0x85845dd1, 21)
    STEP(I, a, b, c, d, GET(8), 0x6fa87e4f, 6)
    STEP(I, d, a, b, c, GET(15), 0xfe2ce6e0, 10)
    STEP(I, c, d, a, b, GET(6), 0xa3014314, 15)
    STEP(I, b, c, d, a, GET(13), 0x4e0811a1, 21)
    STEP(I, a, b, c, d, GET(4), 0xf7537e82, 6)
    STEP(I, d, a, b, c, GET(11), 0xbd3af235, 10)
    STEP(I, c, d, a, b, GET(2), 0x2ad7d2bb, 15)
    STEP(I, b, c, d, a, GET(9), 0xeb86d391, 21)

    a += saved_a;
    b += saved_b;
    c += saved_c;
    d += saved_d;

    ptr += 64;
  } while (Size -= 64);

  this->a = a;
  this->b = b;
  this->c = c;
  this->d = d;

  return ptr;
}

MD5::MD5() = default;

/// Incrementally add the bytes in \p Data to the hash.
void MD5::update(ArrayRef<uint8_t> Data) {
  MD5_u32plus saved_lo;
  unsigned long used, free;
  const uint8_t *Ptr = Data.data();
  unsigned long Size = Data.size();

  saved_lo = lo;
  if ((lo = (saved_lo + Size) & 0x1fffffff) < saved_lo)
    hi++;
  hi += Size >> 29;

  used = saved_lo & 0x3f;

  if (used) {
    free = 64 - used;

    if (Size < free) {
      memcpy(&buffer[used], Ptr, Size);
      return;
    }

    memcpy(&buffer[used], Ptr, free);
    Ptr = Ptr + free;
    Size -= free;
    body(makeArrayRef(buffer, 64));
  }

  if (Size >= 64) {
    Ptr = body(makeArrayRef(Ptr, Size & ~(unsigned long) 0x3f));
    Size &= 0x3f;
  }

  memcpy(buffer, Ptr, Size);
}

/// Add the bytes in the StringRef \p Str to the hash.
// Note that this isn't a string and so this won't include any trailing NULL
// bytes.
void MD5::update(StringRef Str) {
  ArrayRef<uint8_t> SVal((const uint8_t *)Str.data(), Str.size());
  update(SVal);
}

/// Finish the hash and place the resulting hash into \p result.
/// \param Result is assumed to be a minimum of 16-bytes in size.
void MD5::final(MD5Result &Result) {
  unsigned long used, free;

  used = lo & 0x3f;

  buffer[used++] = 0x80;

  free = 64 - used;

  if (free < 8) {
    memset(&buffer[used], 0, free);
    body(makeArrayRef(buffer, 64));
    used = 0;
    free = 64;
  }

  memset(&buffer[used], 0, free - 8);

  lo <<= 3;
  support::endian::write32le(&buffer[56], lo);
  support::endian::write32le(&buffer[60], hi);

  body(makeArrayRef(buffer, 64));

  support::endian::write32le(&Result[0], a);
  support::endian::write32le(&Result[4], b);
  support::endian::write32le(&Result[8], c);
  support::endian::write32le(&Result[12], d);
}

SmallString<32> MD5::MD5Result::digest() const {
  SmallString<32> Str;
  raw_svector_ostream Res(Str);
  for (int i = 0; i < 16; ++i)
    Res << format("%.2x", Bytes[i]);
  return Str;
}

void MD5::stringifyResult(MD5Result &Result, SmallString<32> &Str) {
  Str = Result.digest();
}

std::array<uint8_t, 16> MD5::hash(ArrayRef<uint8_t> Data) {
  MD5 Hash;
  Hash.update(Data);
  MD5::MD5Result Res;
  Hash.final(Res);

  return Res;
}
