/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 10, 2023.
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
#include <stddef.h>
#include <stdint.h>
#ifdef WEBRTC_WEBKIT_BUILD
#include <stdlib.h>
#endif

#include "absl/strings/string_view.h"
#ifdef WEBRTC_WEBKIT_BUILD
#include "api/jsep.h"
#endif
#include "pc/test/integration_test_helpers.h"

namespace webrtc {

class FuzzerTest : public PeerConnectionIntegrationBaseTest {
 public:
  FuzzerTest()
      : PeerConnectionIntegrationBaseTest(SdpSemantics::kUnifiedPlan) {}

#ifdef WEBRTC_WEBKIT_BUILD
  void RunNegotiateCycle(SdpType sdpType, absl::string_view message) {
#else
  void RunNegotiateCycle(absl::string_view message) {
#endif
    CreatePeerConnectionWrappers();
    // Note - we do not do test.ConnectFakeSignaling(); all signals
    // generated are discarded.

    auto srd_observer =
        rtc::make_ref_counted<FakeSetRemoteDescriptionObserver>();

#ifdef WEBRTC_WEBKIT_BUILD
    std::unique_ptr<SessionDescriptionInterface> sdp(
        CreateSessionDescription(sdpType, std::string(message)));
#else
    SdpParseError error;
    std::unique_ptr<SessionDescriptionInterface> sdp(
        CreateSessionDescription("offer", std::string(message), &error));
#endif
    caller()->pc()->SetRemoteDescription(std::move(sdp), srd_observer);
    // Wait a short time for observer to be called. Timeout is short
    // because the fuzzer should be trying many branches.
    EXPECT_TRUE_WAIT(srd_observer->called(), 100);

    // If set-remote-description was successful, try to answer.
    auto sld_observer =
        rtc::make_ref_counted<FakeSetLocalDescriptionObserver>();
    if (srd_observer->error().ok()) {
      caller()->pc()->SetLocalDescription(sld_observer);
      EXPECT_TRUE_WAIT(sld_observer->called(), 100);
    }
#if !defined(WEBRTC_WEBKIT_BUILD)
    // If there is an EXPECT failure, die here.
    RTC_CHECK(!HasFailure());
#endif // !defined(WEBRTC_WEBKIT_BUILD)
  }

  // This test isn't using the test definition macros, so we have to
  // define the TestBody() function, even though we don't need it.
  void TestBody() override {}
};

void FuzzOneInput(const uint8_t* data, size_t size) {
#ifdef WEBRTC_WEBKIT_BUILD
  uint8_t* newData = const_cast<uint8_t*>(data);
  size_t newSize = size;
  uint8_t type = 0;

  if (const char* var = getenv("SDP_TYPE")) {
    if (size > 16384) {
      return;
    }
    type = atoi(var);
  } else {
    if (size < 1 || size > 16385) {
      return;
    }
    type = data[0];
    newSize = size - 1;
    newData = reinterpret_cast<uint8_t*>(malloc(newSize));
    if (!newData)
      return;
    memcpy(newData, &data[1], newSize);
  }

  SdpType sdpType = SdpType::kOffer;
  switch (type % 4) {
    case 0: sdpType = SdpType::kOffer; break;
    case 1: sdpType = SdpType::kPrAnswer; break;
    case 2: sdpType = SdpType::kAnswer; break;
    case 3: sdpType = SdpType::kRollback; break;
  }
#else
  if (size > 16384) {
    return;
  }
#endif

  FuzzerTest test;
#ifdef WEBRTC_WEBKIT_BUILD
  test.RunNegotiateCycle(
      sdpType,
      absl::string_view(reinterpret_cast<const char*>(newData), newSize));

  if (newData != data)
      free(newData);
#else
  test.RunNegotiateCycle(
      absl::string_view(reinterpret_cast<const char*>(data), size));
#endif
}

}  // namespace webrtc
