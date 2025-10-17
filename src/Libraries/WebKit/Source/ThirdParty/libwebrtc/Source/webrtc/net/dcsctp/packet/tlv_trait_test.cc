/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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
#include "net/dcsctp/packet/tlv_trait.h"

#include <vector>

#include "api/array_view.h"
#include "rtc_base/buffer.h"
#include "rtc_base/checks.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {
using ::testing::ElementsAre;
using ::testing::SizeIs;

struct OneByteTypeConfig {
  static constexpr int kTypeSizeInBytes = 1;
  static constexpr int kType = 0x49;
  static constexpr size_t kHeaderSize = 12;
  static constexpr int kVariableLengthAlignment = 4;
};

class OneByteChunk : public TLVTrait<OneByteTypeConfig> {
 public:
  static constexpr size_t kVariableSize = 4;

  void SerializeTo(std::vector<uint8_t>& out) {
    BoundedByteWriter<OneByteTypeConfig::kHeaderSize> writer =
        AllocateTLV(out, kVariableSize);
    writer.Store32<4>(0x01020304);
    writer.Store16<8>(0x0506);
    writer.Store16<10>(0x0708);

    uint8_t variable_data[kVariableSize] = {0xDE, 0xAD, 0xBE, 0xEF};
    writer.CopyToVariableData(rtc::ArrayView<const uint8_t>(variable_data));
  }

  static std::optional<BoundedByteReader<OneByteTypeConfig::kHeaderSize>> Parse(
      rtc::ArrayView<const uint8_t> data) {
    return ParseTLV(data);
  }
};

TEST(TlvDataTest, CanWriteOneByteTypeTlvs) {
  std::vector<uint8_t> out;
  OneByteChunk().SerializeTo(out);

  EXPECT_THAT(out, SizeIs(OneByteTypeConfig::kHeaderSize +
                          OneByteChunk::kVariableSize));
  EXPECT_THAT(out, ElementsAre(0x49, 0x00, 0x00, 0x10, 0x01, 0x02, 0x03, 0x04,
                               0x05, 0x06, 0x07, 0x08, 0xDE, 0xAD, 0xBE, 0xEF));
}

TEST(TlvDataTest, CanReadOneByteTypeTlvs) {
  uint8_t data[] = {0x49, 0x00, 0x00, 0x10, 0x01, 0x02, 0x03, 0x04,
                    0x05, 0x06, 0x07, 0x08, 0xDE, 0xAD, 0xBE, 0xEF};

  std::optional<BoundedByteReader<OneByteTypeConfig::kHeaderSize>> reader =
      OneByteChunk::Parse(data);
  ASSERT_TRUE(reader.has_value());
  EXPECT_EQ(reader->Load32<4>(), 0x01020304U);
  EXPECT_EQ(reader->Load16<8>(), 0x0506U);
  EXPECT_EQ(reader->Load16<10>(), 0x0708U);
  EXPECT_THAT(reader->variable_data(), ElementsAre(0xDE, 0xAD, 0xBE, 0xEF));
}

struct TwoByteTypeConfig {
  static constexpr int kTypeSizeInBytes = 2;
  static constexpr int kType = 31337;
  static constexpr size_t kHeaderSize = 8;
  static constexpr int kVariableLengthAlignment = 2;
};

class TwoByteChunk : public TLVTrait<TwoByteTypeConfig> {
 public:
  static constexpr size_t kVariableSize = 8;

  void SerializeTo(std::vector<uint8_t>& out) {
    BoundedByteWriter<TwoByteTypeConfig::kHeaderSize> writer =
        AllocateTLV(out, kVariableSize);
    writer.Store32<4>(0x01020304U);

    uint8_t variable_data[] = {0x05, 0x06, 0x07, 0x08, 0xDE, 0xAD, 0xBE, 0xEF};
    writer.CopyToVariableData(rtc::ArrayView<const uint8_t>(variable_data));
  }

  static std::optional<BoundedByteReader<TwoByteTypeConfig::kHeaderSize>> Parse(
      rtc::ArrayView<const uint8_t> data) {
    return ParseTLV(data);
  }
};

TEST(TlvDataTest, CanWriteTwoByteTypeTlvs) {
  std::vector<uint8_t> out;

  TwoByteChunk().SerializeTo(out);

  EXPECT_THAT(out, SizeIs(TwoByteTypeConfig::kHeaderSize +
                          TwoByteChunk::kVariableSize));
  EXPECT_THAT(out, ElementsAre(0x7A, 0x69, 0x00, 0x10, 0x01, 0x02, 0x03, 0x04,
                               0x05, 0x06, 0x07, 0x08, 0xDE, 0xAD, 0xBE, 0xEF));
}

TEST(TlvDataTest, CanReadTwoByteTypeTlvs) {
  uint8_t data[] = {0x7A, 0x69, 0x00, 0x10, 0x01, 0x02, 0x03, 0x04,
                    0x05, 0x06, 0x07, 0x08, 0xDE, 0xAD, 0xBE, 0xEF};

  std::optional<BoundedByteReader<TwoByteTypeConfig::kHeaderSize>> reader =
      TwoByteChunk::Parse(data);
  EXPECT_TRUE(reader.has_value());
  EXPECT_EQ(reader->Load32<4>(), 0x01020304U);
  EXPECT_THAT(reader->variable_data(),
              ElementsAre(0x05, 0x06, 0x07, 0x08, 0xDE, 0xAD, 0xBE, 0xEF));
}

TEST(TlvDataTest, CanHandleInvalidLengthSmallerThanFixedSize) {
  // Has 'length=6', which is below the kHeaderSize of 8.
  uint8_t data[] = {0x7A, 0x69, 0x00, 0x06, 0x01, 0x02, 0x03, 0x04};

  EXPECT_FALSE(TwoByteChunk::Parse(data).has_value());
}

}  // namespace
}  // namespace dcsctp
