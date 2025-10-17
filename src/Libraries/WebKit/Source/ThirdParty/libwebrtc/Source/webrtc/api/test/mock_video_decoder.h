/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 8, 2024.
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
#ifndef API_TEST_MOCK_VIDEO_DECODER_H_
#define API_TEST_MOCK_VIDEO_DECODER_H_

#include <cstdint>
#include <optional>

#include "api/video/encoded_image.h"
#include "api/video/video_frame.h"
#include "api/video_codecs/video_decoder.h"
#include "test/gmock.h"

namespace webrtc {

using testing::_;
using testing::Invoke;

class MockDecodedImageCallback : public DecodedImageCallback {
 public:
  MOCK_METHOD(int32_t,
              Decoded,
              (VideoFrame & decoded_image),  // NOLINT
              (override));
  MOCK_METHOD(int32_t,
              Decoded,
              (VideoFrame & decoded_image,  // NOLINT
               int64_t decode_time_ms),
              (override));
  MOCK_METHOD(void,
              Decoded,
              (VideoFrame & decoded_image,  // NOLINT
               std::optional<int32_t> decode_time_ms,
               std::optional<uint8_t> qp),
              (override));
};

class MockVideoDecoder : public VideoDecoder {
 public:
  MockVideoDecoder() {
    // Make `Configure` succeed by default, so that individual tests that
    // verify other methods wouldn't need to stub `Configure`.
    ON_CALL(*this, Configure).WillByDefault(testing::Return(true));

    // TODO(bugs.webrtc.org/15444): Remove once all tests have been migrated to
    // expecting calls Decode without a missing_frames param.
    ON_CALL(*this, Decode(_, _))
        .WillByDefault(Invoke([this](const EncodedImage& input_image,
                                     int64_t render_time_ms) {
          return Decode(input_image, /*missing_frames=*/false, render_time_ms);
        }));
  }

  ~MockVideoDecoder() override { Destruct(); }

  MOCK_METHOD(bool, Configure, (const Settings& settings), (override));
  MOCK_METHOD(int32_t,
              Decode,
              (const EncodedImage& input_image,
               int64_t render_time_ms),
              (override));
  MOCK_METHOD(int32_t,
              Decode,
              (const EncodedImage& input_image,
               bool missing_frames,
               int64_t render_time_ms));
  MOCK_METHOD(int32_t,
              RegisterDecodeCompleteCallback,
              (DecodedImageCallback * callback),
              (override));
  MOCK_METHOD(int32_t, Release, (), (override));

  // Special utility method that allows a test to monitor/verify when
  // destruction of the decoder instance occurs.
  MOCK_METHOD(void, Destruct, (), ());
};

}  // namespace webrtc

#endif  // API_TEST_MOCK_VIDEO_DECODER_H_
