/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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
#ifndef NET_DCSCTP_PACKET_BOUNDED_BYTE_WRITER_H_
#define NET_DCSCTP_PACKET_BOUNDED_BYTE_WRITER_H_

#include <algorithm>

#include "api/array_view.h"

namespace dcsctp {

// TODO(boivie): These generic functions - and possibly this entire class -
// could be a candidate to have added to rtc_base/. They should use compiler
// intrinsics as well.
namespace internal {
// Stores a 8-bit unsigned word at `data`.
inline void StoreBigEndian8(uint8_t* data, uint8_t val) {
  data[0] = val;
}

// Stores a 16-bit unsigned word at `data`.
inline void StoreBigEndian16(uint8_t* data, uint16_t val) {
  data[0] = val >> 8;
  data[1] = val;
}

// Stores a 32-bit unsigned word at `data`.
inline void StoreBigEndian32(uint8_t* data, uint32_t val) {
  data[0] = val >> 24;
  data[1] = val >> 16;
  data[2] = val >> 8;
  data[3] = val;
}
}  // namespace internal

// BoundedByteWriter wraps an ArrayView and divides it into two parts; A fixed
// size - which is the template parameter - and a variable size, which is what
// remains in `data` after the `FixedSize`.
//
// The BoundedByteWriter provides methods to write big endian numbers to the
// FixedSize portion of the buffer, and these are written with static bounds
// checking, to avoid out-of-bounds accesses without a run-time penalty.
//
// The variable sized portion can either be used to create sub-writers, which
// themselves would provide compile-time bounds-checking, or data can be copied
// to it.
template <int FixedSize>
class BoundedByteWriter {
 public:
  explicit BoundedByteWriter(rtc::ArrayView<uint8_t> data) : data_(data) {
    RTC_CHECK(data.size() >= FixedSize);
  }

  template <size_t offset>
  void Store8(uint8_t value) {
    static_assert(offset + sizeof(uint8_t) <= FixedSize, "Out-of-bounds");
    internal::StoreBigEndian8(&data_[offset], value);
  }

  template <size_t offset>
  void Store16(uint16_t value) {
    static_assert(offset + sizeof(uint16_t) <= FixedSize, "Out-of-bounds");
    static_assert((offset % sizeof(uint16_t)) == 0, "Unaligned access");
    internal::StoreBigEndian16(&data_[offset], value);
  }

  template <size_t offset>
  void Store32(uint32_t value) {
    static_assert(offset + sizeof(uint32_t) <= FixedSize, "Out-of-bounds");
    static_assert((offset % sizeof(uint32_t)) == 0, "Unaligned access");
    internal::StoreBigEndian32(&data_[offset], value);
  }

  template <size_t SubSize>
  BoundedByteWriter<SubSize> sub_writer(size_t variable_offset) {
    RTC_CHECK(FixedSize + variable_offset + SubSize <= data_.size());

    return BoundedByteWriter<SubSize>(
        data_.subview(FixedSize + variable_offset, SubSize));
  }

  void CopyToVariableData(rtc::ArrayView<const uint8_t> source) {
    size_t copy_size = std::min(source.size(), data_.size() - FixedSize);
    if (source.data() == nullptr || copy_size == 0) {
      return;
    }
    memcpy(data_.data() + FixedSize, source.data(), copy_size);
  }

 private:
  rtc::ArrayView<uint8_t> data_;
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_BOUNDED_BYTE_WRITER_H_
