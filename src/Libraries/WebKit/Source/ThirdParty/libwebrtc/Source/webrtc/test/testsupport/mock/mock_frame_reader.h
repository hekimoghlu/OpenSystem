/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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
#ifndef TEST_TESTSUPPORT_MOCK_MOCK_FRAME_READER_H_
#define TEST_TESTSUPPORT_MOCK_MOCK_FRAME_READER_H_

#include "api/video/i420_buffer.h"
#include "test/gmock.h"
#include "test/testsupport/frame_reader.h"

namespace webrtc {
namespace test {

class MockFrameReader : public FrameReader {
 public:
  MOCK_METHOD(rtc::scoped_refptr<I420Buffer>, PullFrame, (), (override));
  MOCK_METHOD(rtc::scoped_refptr<I420Buffer>, PullFrame, (int*), (override));
  MOCK_METHOD(rtc::scoped_refptr<I420Buffer>,
              PullFrame,
              (int*, Resolution, Ratio),
              (override));
  MOCK_METHOD(rtc::scoped_refptr<I420Buffer>, ReadFrame, (int), (override));
  MOCK_METHOD(rtc::scoped_refptr<I420Buffer>,
              ReadFrame,
              (int, Resolution),
              (override));
  MOCK_METHOD(int, num_frames, (), (const, override));
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_TESTSUPPORT_MOCK_MOCK_FRAME_READER_H_
