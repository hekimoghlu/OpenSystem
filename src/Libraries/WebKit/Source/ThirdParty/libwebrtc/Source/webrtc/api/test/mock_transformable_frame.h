/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 17, 2021.
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
#ifndef API_TEST_MOCK_TRANSFORMABLE_FRAME_H_
#define API_TEST_MOCK_TRANSFORMABLE_FRAME_H_

#include <stdint.h>

#include <optional>
#include <string>
#include <type_traits>

#include "api/array_view.h"
#include "api/frame_transformer_interface.h"
#include "api/units/timestamp.h"
#include "test/gmock.h"

namespace webrtc {

class MockTransformableFrame : public TransformableFrameInterface {
 public:
  MockTransformableFrame() : TransformableFrameInterface(Passkey()) {}

  MOCK_METHOD(rtc::ArrayView<const uint8_t>, GetData, (), (const, override));
  MOCK_METHOD(void, SetData, (rtc::ArrayView<const uint8_t>), (override));
  MOCK_METHOD(uint8_t, GetPayloadType, (), (const, override));
  MOCK_METHOD(uint32_t, GetSsrc, (), (const, override));
  MOCK_METHOD(uint32_t, GetTimestamp, (), (const, override));
  MOCK_METHOD(void, SetRTPTimestamp, (uint32_t), (override));
  MOCK_METHOD(std::optional<webrtc::Timestamp>,
              GetPresentationTimestamp,
              (),
              (const, override));
  MOCK_METHOD(std::string, GetMimeType, (), (const, override));
};

static_assert(!std::is_abstract_v<MockTransformableFrame>, "");

}  // namespace webrtc

#endif  // API_TEST_MOCK_TRANSFORMABLE_FRAME_H_
