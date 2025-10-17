/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
#include <stdio.h>

#include <string>
#include <vector>

#include "modules/audio_coding/include/audio_coding_module.h"
#include "modules/audio_coding/test/EncodeDecodeTest.h"
#include "modules/audio_coding/test/PacketLossTest.h"
#include "modules/audio_coding/test/TestAllCodecs.h"
#include "modules/audio_coding/test/TestRedFec.h"
#include "modules/audio_coding/test/TestStereo.h"
#include "modules/audio_coding/test/TestVADDTX.h"
#include "modules/audio_coding/test/opus_test.h"
#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

TEST(AudioCodingModuleTest, TestAllCodecs) {
  webrtc::TestAllCodecs().Perform();
}

#if defined(WEBRTC_ANDROID)
TEST(AudioCodingModuleTest, DISABLED_TestEncodeDecode) {
#else
TEST(AudioCodingModuleTest, TestEncodeDecode) {
#endif
  webrtc::EncodeDecodeTest().Perform();
}

TEST(AudioCodingModuleTest, TestRedFec) {
  webrtc::TestRedFec().Perform();
}

// Disabled on ios as flaky, see https://crbug.com/webrtc/7057
#if defined(WEBRTC_ANDROID) || defined(WEBRTC_IOS)
TEST(AudioCodingModuleTest, DISABLED_TestStereo) {
#else
TEST(AudioCodingModuleTest, TestStereo) {
#endif
  webrtc::TestStereo().Perform();
}

TEST(AudioCodingModuleTest, TestWebRtcVadDtx) {
  webrtc::TestWebRtcVadDtx().Perform();
}

TEST(AudioCodingModuleTest, TestOpusDtx) {
  webrtc::TestOpusDtx().Perform();
}

// Disabled on ios as flaky, see https://crbug.com/webrtc/7057
#if defined(WEBRTC_IOS)
TEST(AudioCodingModuleTest, DISABLED_TestOpus) {
#else
TEST(AudioCodingModuleTest, TestOpus) {
#endif
  webrtc::OpusTest().Perform();
}

TEST(AudioCodingModuleTest, TestPacketLoss) {
  webrtc::PacketLossTest(1, 10, 10, 1).Perform();
}

TEST(AudioCodingModuleTest, TestPacketLossBurst) {
  webrtc::PacketLossTest(1, 10, 10, 2).Perform();
}

// Disabled on ios as flake, see https://crbug.com/webrtc/7057
#if defined(WEBRTC_IOS)
TEST(AudioCodingModuleTest, DISABLED_TestPacketLossStereo) {
#else
TEST(AudioCodingModuleTest, TestPacketLossStereo) {
#endif
  webrtc::PacketLossTest(2, 10, 10, 1).Perform();
}

// Disabled on ios as flake, see https://crbug.com/webrtc/7057
#if defined(WEBRTC_IOS)
TEST(AudioCodingModuleTest, DISABLED_TestPacketLossStereoBurst) {
#else
TEST(AudioCodingModuleTest, TestPacketLossStereoBurst) {
#endif
  webrtc::PacketLossTest(2, 10, 10, 2).Perform();
}

// The full API test is too long to run automatically on bots, but can be used
// for offline testing. User interaction is needed.
#ifdef ACM_TEST_FULL_API
TEST(AudioCodingModuleTest, TestAPI) {
  webrtc::APITest().Perform();
}
#endif
