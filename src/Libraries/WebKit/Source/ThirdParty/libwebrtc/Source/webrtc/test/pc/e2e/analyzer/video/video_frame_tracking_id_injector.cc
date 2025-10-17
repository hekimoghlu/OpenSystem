/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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
#include "test/pc/e2e/analyzer/video/video_frame_tracking_id_injector.h"

#include "absl/memory/memory.h"
#include "api/video/encoded_image.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace webrtc_pc_e2e {

EncodedImage VideoFrameTrackingIdInjector::InjectData(
    uint16_t id,
    bool unused_discard,
    const EncodedImage& source) {
  RTC_CHECK(!unused_discard);
  EncodedImage out = source;
  out.SetVideoFrameTrackingId(id);
  return out;
}

EncodedImageExtractionResult VideoFrameTrackingIdInjector::ExtractData(
    const EncodedImage& source) {
  return EncodedImageExtractionResult{source.VideoFrameTrackingId(), source,
                                      /*discard=*/false};
}

}  // namespace webrtc_pc_e2e
}  // namespace webrtc
