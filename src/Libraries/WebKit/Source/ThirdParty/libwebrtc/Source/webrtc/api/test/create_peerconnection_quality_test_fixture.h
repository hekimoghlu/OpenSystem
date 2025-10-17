/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 18, 2023.
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
#ifndef API_TEST_CREATE_PEERCONNECTION_QUALITY_TEST_FIXTURE_H_
#define API_TEST_CREATE_PEERCONNECTION_QUALITY_TEST_FIXTURE_H_

#include <memory>
#include <string>

#include "api/test/audio_quality_analyzer_interface.h"
#include "api/test/peerconnection_quality_test_fixture.h"
#include "api/test/time_controller.h"
#include "api/test/video_quality_analyzer_interface.h"

namespace webrtc {
namespace webrtc_pc_e2e {

// API is in development. Can be changed/removed without notice.

// Create test fixture to establish test call between Alice and Bob.
// During the test Alice will be caller and Bob will answer the call.
// `test_case_name` is a name of test case, that will be used for all metrics
// reporting.
// `time_controller` is used to manage all rtc::Thread's and TaskQueue
// instances. Instance of `time_controller` have to outlive created fixture.
// Returns a non-null PeerConnectionE2EQualityTestFixture instance.
std::unique_ptr<PeerConnectionE2EQualityTestFixture>
CreatePeerConnectionE2EQualityTestFixture(
    std::string test_case_name,
    TimeController& time_controller,
    std::unique_ptr<AudioQualityAnalyzerInterface> audio_quality_analyzer,
    std::unique_ptr<VideoQualityAnalyzerInterface> video_quality_analyzer);

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // API_TEST_CREATE_PEERCONNECTION_QUALITY_TEST_FIXTURE_H_
