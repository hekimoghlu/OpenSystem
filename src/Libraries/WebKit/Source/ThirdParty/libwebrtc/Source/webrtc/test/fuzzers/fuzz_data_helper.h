/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 5, 2024.
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
#ifndef TEST_FUZZERS_FUZZ_DATA_HELPER_H_
#define TEST_FUZZERS_FUZZ_DATA_HELPER_H_

#include <limits>

#include "api/array_view.h"
#include "modules/rtp_rtcp/source/byte_io.h"

namespace webrtc {
namespace test {

// Helper class to take care of the fuzzer input, read from it, and keep track
// of when the end of the data has been reached.
class FuzzDataHelper {
 public:
  explicit FuzzDataHelper(rtc::ArrayView<const uint8_t> data);

  // Returns true if n bytes can be read.
  bool CanReadBytes(size_t n) const { return data_ix_ + n <= data_.size(); }

  // Reads and returns data of type T.
  template <typename T>
  T Read() {
    RTC_CHECK(CanReadBytes(sizeof(T)));
    T x = ByteReader<T>::ReadLittleEndian(&data_[data_ix_]);
    data_ix_ += sizeof(T);
    return x;
  }

  // Reads and returns data of type T. Returns default_value if not enough
  // fuzzer input remains to read a T.
  template <typename T>
  T ReadOrDefaultValue(T default_value) {
    if (!CanReadBytes(sizeof(T))) {
      return default_value;
    }
    return Read<T>();
  }

  // Like ReadOrDefaultValue, but replaces the value 0 with default_value.
  template <typename T>
  T ReadOrDefaultValueNotZero(T default_value) {
    static_assert(std::is_integral<T>::value, "");
    T x = ReadOrDefaultValue(default_value);
    return x == 0 ? default_value : x;
  }

  // Returns one of the elements from the provided input array. The selection
  // is based on the fuzzer input data. If not enough fuzzer data is available,
  // the method will return the first element in the input array. The reason for
  // not flagging this as an error is to allow the method to be called from
  // class constructors, and in constructors we typically do not handle
  // errors. The code will work anyway, and the fuzzer will likely see that
  // providing more data will actually make this method return something else.
  template <typename T, size_t N>
  T SelectOneOf(const T (&select_from)[N]) {
    static_assert(N <= std::numeric_limits<uint8_t>::max(), "");
    // Read an index between 0 and select_from.size() - 1 from the fuzzer data.
    uint8_t index = ReadOrDefaultValue<uint8_t>(0) % N;
    return select_from[index];
  }

  rtc::ArrayView<const uint8_t> ReadByteArray(size_t bytes) {
    if (!CanReadBytes(bytes)) {
      return rtc::ArrayView<const uint8_t>(nullptr, 0);
    }
    const size_t index_to_return = data_ix_;
    data_ix_ += bytes;
    return data_.subview(index_to_return, bytes);
  }

  // If sizeof(T) > BytesLeft then the remaining bytes will be used and the rest
  // of the object will be zero initialized.
  template <typename T>
  void CopyTo(T* object) {
    memset(object, 0, sizeof(T));

    size_t bytes_to_copy = std::min(BytesLeft(), sizeof(T));
    memcpy(object, data_.data() + data_ix_, bytes_to_copy);
    data_ix_ += bytes_to_copy;
  }

  size_t BytesRead() const { return data_ix_; }

  size_t BytesLeft() const { return data_.size() - data_ix_; }

 private:
  rtc::ArrayView<const uint8_t> data_;
  size_t data_ix_ = 0;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_FUZZERS_FUZZ_DATA_HELPER_H_
