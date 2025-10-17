/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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
#ifndef API_TEST_SIMULCAST_TEST_FIXTURE_H_
#define API_TEST_SIMULCAST_TEST_FIXTURE_H_

namespace webrtc {
namespace test {

class SimulcastTestFixture {
 public:
  virtual ~SimulcastTestFixture() = default;

  virtual void TestKeyFrameRequestsOnAllStreams() = 0;
  virtual void TestKeyFrameRequestsOnSpecificStreams() = 0;
  virtual void TestPaddingAllStreams() = 0;
  virtual void TestPaddingTwoStreams() = 0;
  virtual void TestPaddingTwoStreamsOneMaxedOut() = 0;
  virtual void TestPaddingOneStream() = 0;
  virtual void TestPaddingOneStreamTwoMaxedOut() = 0;
  virtual void TestSendAllStreams() = 0;
  virtual void TestDisablingStreams() = 0;
  virtual void TestActiveStreams() = 0;
  virtual void TestSwitchingToOneStream() = 0;
  virtual void TestSwitchingToOneOddStream() = 0;
  virtual void TestSwitchingToOneSmallStream() = 0;
  virtual void TestSpatioTemporalLayers333PatternEncoder() = 0;
  virtual void TestSpatioTemporalLayers321PatternEncoder() = 0;
  virtual void TestStrideEncodeDecode() = 0;
  virtual void TestDecodeWidthHeightSet() = 0;
  virtual void
  TestEncoderInfoForDefaultTemporalLayerProfileHasFpsAllocation() = 0;
};

}  // namespace test
}  // namespace webrtc

#endif  // API_TEST_SIMULCAST_TEST_FIXTURE_H_
