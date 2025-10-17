/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 15, 2024.
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
#ifndef LLVM_SUPPORT_MD5_H
#define LLVM_SUPPORT_MD5_H

#include <IndexStoreDB_LLVMSupport/toolchain_ADT_SmallString.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_StringRef.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_Endian.h>
#include <array>
#include <cstdint>

namespace toolchain {

template <typename T> class ArrayRef;

class MD5 {
  // Any 32-bit or wider unsigned integer data type will do.
  typedef uint32_t MD5_u32plus;

  MD5_u32plus a = 0x67452301;
  MD5_u32plus b = 0xefcdab89;
  MD5_u32plus c = 0x98badcfe;
  MD5_u32plus d = 0x10325476;
  MD5_u32plus hi = 0;
  MD5_u32plus lo = 0;
  uint8_t buffer[64];
  MD5_u32plus block[16];

public:
  struct MD5Result {
    std::array<uint8_t, 16> Bytes;

    operator std::array<uint8_t, 16>() const { return Bytes; }

    const uint8_t &operator[](size_t I) const { return Bytes[I]; }
    uint8_t &operator[](size_t I) { return Bytes[I]; }

    SmallString<32> digest() const;

    uint64_t low() const {
      // Our MD5 implementation returns the result in little endian, so the low
      // word is first.
      using namespace support;
      return endian::read<uint64_t, little, unaligned>(Bytes.data());
    }

    uint64_t high() const {
      using namespace support;
      return endian::read<uint64_t, little, unaligned>(Bytes.data() + 8);
    }
    std::pair<uint64_t, uint64_t> words() const {
      using namespace support;
      return std::make_pair(high(), low());
    }
  };

  MD5();

  /// Updates the hash for the byte stream provided.
  void update(ArrayRef<uint8_t> Data);

  /// Updates the hash for the StringRef provided.
  void update(StringRef Str);

  /// Finishes off the hash and puts the result in result.
  void final(MD5Result &Result);

  /// Translates the bytes in \p Res to a hex string that is
  /// deposited into \p Str. The result will be of length 32.
  static void stringifyResult(MD5Result &Result, SmallString<32> &Str);

  /// Computes the hash for a given bytes.
  static std::array<uint8_t, 16> hash(ArrayRef<uint8_t> Data);

private:
  const uint8_t *body(ArrayRef<uint8_t> Data);
};

inline bool operator==(const MD5::MD5Result &LHS, const MD5::MD5Result &RHS) {
  return LHS.Bytes == RHS.Bytes;
}

/// Helper to compute and return lower 64 bits of the given string's MD5 hash.
inline uint64_t MD5Hash(StringRef Str) {
  using namespace support;

  MD5 Hash;
  Hash.update(Str);
  MD5::MD5Result Result;
  Hash.final(Result);
  // Return the least significant word.
  return Result.low();
}

} // end namespace toolchain

#endif // LLVM_SUPPORT_MD5_H
