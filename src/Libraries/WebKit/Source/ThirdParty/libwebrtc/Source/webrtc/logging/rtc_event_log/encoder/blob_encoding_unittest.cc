/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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
#include "logging/rtc_event_log/encoder/blob_encoding.h"

#include <cstddef>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "logging/rtc_event_log/encoder/var_int.h"
#include "rtc_base/checks.h"
#include "test/gtest.h"

using CharT = std::string::value_type;

namespace webrtc {

namespace {

void TestEncodingAndDecoding(const std::vector<std::string>& blobs) {
  RTC_DCHECK(!blobs.empty());

  const std::string encoded = EncodeBlobs(blobs);
  ASSERT_FALSE(encoded.empty());

  const std::vector<absl::string_view> decoded =
      DecodeBlobs(encoded, blobs.size());

  ASSERT_EQ(decoded.size(), blobs.size());
  for (size_t i = 0; i < decoded.size(); ++i) {
    ASSERT_EQ(decoded[i], blobs[i]);
  }
}

void TestGracefulErrorHandling(absl::string_view encoded_blobs,
                               size_t num_of_blobs) {
  const std::vector<absl::string_view> decoded =
      DecodeBlobs(encoded_blobs, num_of_blobs);
  EXPECT_TRUE(decoded.empty());
}

}  // namespace

TEST(BlobEncoding, EmptyBlob) {
  TestEncodingAndDecoding({""});
}

TEST(BlobEncoding, SingleCharacterBlob) {
  TestEncodingAndDecoding({"a"});
}

TEST(BlobEncoding, LongBlob) {
  std::string blob = "";
  for (size_t i = 0; i < 100000; ++i) {
    blob += std::to_string(i + 1) + " Mississippi\n";
  }
  TestEncodingAndDecoding({blob});
}

TEST(BlobEncoding, BlobsOfVariousLengths) {
  constexpr size_t kJump = 0xf032d;  // Arbitrary.
  constexpr size_t kMax = 0xffffff;  // Arbitrary.

  std::string blob;
  blob.reserve(kMax);

  for (size_t i = 0; i < kMax; i += kJump) {
    blob.append(kJump, 'x');
    TestEncodingAndDecoding({blob});
  }
}

TEST(BlobEncoding, MultipleBlobs) {
  std::vector<std::string> blobs;
  for (size_t i = 0; i < 100000; ++i) {
    blobs.push_back(std::to_string(i + 1) + " Mississippi\n");
  }
  TestEncodingAndDecoding(blobs);
}

TEST(BlobEncoding, DecodeBlobsHandlesErrorsGracefullyEmptyInput) {
  TestGracefulErrorHandling("", 1);
}

TEST(BlobEncoding, DecodeBlobsHandlesErrorsGracefullyZeroBlobs) {
  const std::string encoded = EncodeBlobs({"a"});
  ASSERT_FALSE(encoded.empty());
  TestGracefulErrorHandling(encoded, 0);
}

TEST(BlobEncoding, DecodeBlobsHandlesErrorsGracefullyBlobLengthTooSmall) {
  std::string encoded = EncodeBlobs({"ab"});
  ASSERT_FALSE(encoded.empty());
  ASSERT_EQ(encoded[0], 0x02);
  encoded[0] = 0x01;
  TestGracefulErrorHandling(encoded, 1);
}

TEST(BlobEncoding, DecodeBlobsHandlesErrorsGracefullyBlobLengthTooLarge) {
  std::string encoded = EncodeBlobs({"a"});
  ASSERT_FALSE(encoded.empty());
  ASSERT_EQ(encoded[0], 0x01);
  encoded[0] = 0x02;
  TestGracefulErrorHandling(encoded, 1);
}

TEST(BlobEncoding,
     DecodeBlobsHandlesErrorsGracefullyNumberOfBlobsIncorrectlyHigh) {
  const std::vector<std::string> blobs = {"a", "b"};
  const std::string encoded = EncodeBlobs(blobs);
  // Test focus - two empty strings encoded, but DecodeBlobs() told way more
  // blobs are in the strings than could be expected.
  TestGracefulErrorHandling(encoded, 1000);

  // Test sanity - show that DecodeBlobs() would have worked if it got the
  // correct input.
  TestEncodingAndDecoding(blobs);
}

TEST(BlobEncoding, DecodeBlobsHandlesErrorsGracefullyDefectiveVarInt) {
  std::string defective_varint;
  for (size_t i = 0; i < kMaxVarIntLengthBytes; ++i) {
    ASSERT_LE(kMaxVarIntLengthBytes, 0xffu);
    defective_varint += static_cast<CharT>(static_cast<size_t>(0x80u) | i);
  }
  defective_varint += 0x01u;

  const std::string defective_encoded = defective_varint + "whatever";

  TestGracefulErrorHandling(defective_encoded, 1);
}

TEST(BlobEncoding, DecodeBlobsHandlesErrorsGracefullyLengthSumWrapAround) {
  std::string max_size_varint;
  for (size_t i = 0; i < kMaxVarIntLengthBytes - 1; ++i) {
    max_size_varint += 0xffu;
  }
  max_size_varint += 0x7fu;

  const std::string defective_encoded =
      max_size_varint + max_size_varint + "whatever";

  TestGracefulErrorHandling(defective_encoded, 2);
}

}  // namespace webrtc
