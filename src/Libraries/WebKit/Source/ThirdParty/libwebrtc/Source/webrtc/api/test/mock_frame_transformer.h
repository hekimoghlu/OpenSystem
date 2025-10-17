/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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
#ifndef API_TEST_MOCK_FRAME_TRANSFORMER_H_
#define API_TEST_MOCK_FRAME_TRANSFORMER_H_

#include <cstdint>
#include <memory>

#include "api/frame_transformer_interface.h"
#include "api/scoped_refptr.h"
#include "test/gmock.h"

namespace webrtc {

class MockFrameTransformer : public FrameTransformerInterface {
 public:
  MOCK_METHOD(void,
              Transform,
              (std::unique_ptr<TransformableFrameInterface>),
              (override));
  MOCK_METHOD(void,
              RegisterTransformedFrameCallback,
              (rtc::scoped_refptr<TransformedFrameCallback>),
              (override));
  MOCK_METHOD(void,
              RegisterTransformedFrameSinkCallback,
              (rtc::scoped_refptr<TransformedFrameCallback>, uint32_t),
              (override));
  MOCK_METHOD(void, UnregisterTransformedFrameCallback, (), (override));
  MOCK_METHOD(void,
              UnregisterTransformedFrameSinkCallback,
              (uint32_t),
              (override));
};

}  // namespace webrtc

#endif  // API_TEST_MOCK_FRAME_TRANSFORMER_H_
