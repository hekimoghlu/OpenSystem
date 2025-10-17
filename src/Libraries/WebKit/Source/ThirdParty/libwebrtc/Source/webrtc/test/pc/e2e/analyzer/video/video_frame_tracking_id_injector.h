/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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
#ifndef TEST_PC_E2E_ANALYZER_VIDEO_VIDEO_FRAME_TRACKING_ID_INJECTOR_H_
#define TEST_PC_E2E_ANALYZER_VIDEO_VIDEO_FRAME_TRACKING_ID_INJECTOR_H_

#include <cstdint>

#include "api/video/encoded_image.h"
#include "test/pc/e2e/analyzer/video/encoded_image_data_injector.h"

namespace webrtc {
namespace webrtc_pc_e2e {

// This injector sets and retrieves the provided id in the EncodedImage
// video_frame_tracking_id field. This is only possible with the RTP header
// extension VideoFrameTrackingIdExtension that will propagate the input
// tracking id to the received EncodedImage. This RTP header extension is
// enabled with the field trial WebRTC-VideoFrameTrackingIdAdvertised
// (http://www.webrtc.org/experiments/rtp-hdrext/video-frame-tracking-id).
//
// Note that this injector doesn't allow to discard frames.
class VideoFrameTrackingIdInjector : public EncodedImageDataPropagator {
 public:
  EncodedImage InjectData(uint16_t id,
                          bool unused_discard,
                          const EncodedImage& source) override;

  EncodedImageExtractionResult ExtractData(const EncodedImage& source) override;

  void Start(int) override {}
  void AddParticipantInCall() override {}
  void RemoveParticipantInCall() override {}
};

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // TEST_PC_E2E_ANALYZER_VIDEO_VIDEO_FRAME_TRACKING_ID_INJECTOR_H_
