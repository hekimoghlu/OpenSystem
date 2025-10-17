/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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
#ifndef NET_DCSCTP_PACKET_BOUNDED_BYTE_READER_H_
#define NET_DCSCTP_PACKET_BOUNDED_BYTE_READER_H_

#include <cstdint>

#include "api/array_view.h"

namespace dcsctp {

// TODO(boivie): These generic functions - and possibly this entire class -
// could be a candidate to have added to rtc_base/. They should use compiler
// intrinsics as well.
namespace internal {
// Loads a 8-bit unsigned word at `data`.
inline uint8_t LoadBigEndian8(const uint8_t* data) {
  return data[0];
}

// Loads a 16-bit unsigned word at `data`.
inline uint16_t LoadBigEndian16(const uint8_t* data) {
  return (data[0] << 8) | data[1];
}

// Loads a 32-bit unsigned word at `data`.
inline uint32_t LoadBigEndian32(const uint8_t* data) {
  return (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
}
}  // namespace internal

// BoundedByteReader wraps an ArrayView and divides it into two parts; A fixed
// size - which is the template parameter - and a variable size, which is what
// remains in `data` after the `FixedSize`.
//
// The BoundedByteReader provides methods to load/read big endian numbers from
// the FixedSize portion of the buffer, and these are read with static bounds
// checking, to avoid out-of-bounds accesses without a run-time penalty.
//
// The variable sized portion can either be used to create sub-readers, which
// themselves would provide compile-time bounds-checking, or the entire variable
// sized portion can be retrieved as an ArrayView.
template <int FixedSize>
class BoundedByteReader {
 public:
  explicit BoundedByteReader(rtc::ArrayView<const uint8_t> data) : data_(data) {
    RTC_CHECK(data.size() >= FixedSize);
  }

  template <size_t offset>
  uint8_t Load8() const {
    static_assert(offset + sizeof(uint8_t) <= FixedSize, "Out-of-bounds");
    return internal::LoadBigEndian8(&data_[offset]);
  }

  template <size_t offset>
  uint16_t Load16() const {
    static_assert(offset + sizeof(uint16_t) <= FixedSize, "Out-of-bounds");
    static_assert((offset % sizeof(uint16_t)) == 0, "Unaligned access");
    return internal::LoadBigEndian16(&data_[offset]);
  }

  template <size_t offset>
  uint32_t Load32() const {
    static_assert(offset + sizeof(uint32_t) <= FixedSize, "Out-of-bounds");
    static_assert((offset % sizeof(uint32_t)) == 0, "Unaligned access");
    return internal::LoadBigEndian32(&data_[offset]);
  }

  template <size_t SubSize>
  BoundedByteReader<SubSize> sub_reader(size_t variable_offset) const {
    RTC_CHECK(FixedSize + variable_offset + SubSize <= data_.size());

    rtc::ArrayView<const uint8_t> sub_span =
        data_.subview(FixedSize + variable_offset, SubSize);
    return BoundedByteReader<SubSize>(sub_span);
  }

  size_t variable_data_size() const { return data_.size() - FixedSize; }

  rtc::ArrayView<const uint8_t> variable_data() const {
    return data_.subview(FixedSize, data_.size() - FixedSize);
  }

 private:
  const rtc::ArrayView<const uint8_t> data_;
};

}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_BOUNDED_BYTE_READER_H_
