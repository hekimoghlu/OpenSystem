/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 4, 2022.
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
#ifndef API_VIDEO_TEST_MOCK_RECORDABLE_ENCODED_FRAME_H_
#define API_VIDEO_TEST_MOCK_RECORDABLE_ENCODED_FRAME_H_

#include "api/video/recordable_encoded_frame.h"
#include "test/gmock.h"

namespace webrtc {
class MockRecordableEncodedFrame : public RecordableEncodedFrame {
 public:
  MOCK_METHOD(rtc::scoped_refptr<const EncodedImageBufferInterface>,
              encoded_buffer,
              (),
              (const, override));
  MOCK_METHOD(std::optional<webrtc::ColorSpace>,
              color_space,
              (),
              (const, override));
  MOCK_METHOD(VideoCodecType, codec, (), (const, override));
  MOCK_METHOD(bool, is_key_frame, (), (const, override));
  MOCK_METHOD(EncodedResolution, resolution, (), (const, override));
  MOCK_METHOD(Timestamp, render_time, (), (const, override));
};
}  // namespace webrtc
#endif  // API_VIDEO_TEST_MOCK_RECORDABLE_ENCODED_FRAME_H_
