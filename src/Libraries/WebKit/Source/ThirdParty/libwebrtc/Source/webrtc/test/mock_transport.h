/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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
#ifndef TEST_MOCK_TRANSPORT_H_
#define TEST_MOCK_TRANSPORT_H_

#include "api/call/transport.h"
#include "test/gmock.h"

namespace webrtc {

class MockTransport : public Transport {
 public:
  MockTransport();
  ~MockTransport();

  MOCK_METHOD(bool,
              SendRtp,
              (rtc::ArrayView<const uint8_t>, const PacketOptions&),
              (override));
  MOCK_METHOD(bool, SendRtcp, (rtc::ArrayView<const uint8_t>), (override));
};

}  // namespace webrtc

#endif  // TEST_MOCK_TRANSPORT_H_
