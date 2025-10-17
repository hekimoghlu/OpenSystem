/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 29, 2025.
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
#ifndef API_TEST_MOCK_SESSION_DESCRIPTION_INTERFACE_H_
#define API_TEST_MOCK_SESSION_DESCRIPTION_INTERFACE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "api/jsep.h"
#include "test/gmock.h"

namespace webrtc {

class MockSessionDescriptionInterface : public SessionDescriptionInterface {
 public:
  MOCK_METHOD(std::unique_ptr<SessionDescriptionInterface>,
              Clone,
              (),
              (const, override));
  MOCK_METHOD(cricket::SessionDescription*, description, (), (override));
  MOCK_METHOD(const cricket::SessionDescription*,
              description,
              (),
              (const, override));
  MOCK_METHOD(std::string, session_id, (), (const, override));
  MOCK_METHOD(std::string, session_version, (), (const, override));
  MOCK_METHOD(SdpType, GetType, (), (const, override));
  MOCK_METHOD(std::string, type, (), (const, override));
  MOCK_METHOD(bool, AddCandidate, (const IceCandidateInterface*), (override));
  MOCK_METHOD(size_t,
              RemoveCandidates,
              (const std::vector<cricket::Candidate>&),
              (override));
  MOCK_METHOD(size_t, number_of_mediasections, (), (const, override));
  MOCK_METHOD(const IceCandidateCollection*,
              candidates,
              (size_t),
              (const, override));
  MOCK_METHOD(bool, ToString, (std::string*), (const, override));
};

static_assert(!std::is_abstract_v<MockSessionDescriptionInterface>);

}  // namespace webrtc

#endif  // API_TEST_MOCK_SESSION_DESCRIPTION_INTERFACE_H_
