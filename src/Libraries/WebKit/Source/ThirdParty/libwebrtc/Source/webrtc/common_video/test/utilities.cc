/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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
#include "common_video/test/utilities.h"

#include <utility>

namespace webrtc {

HdrMetadata CreateTestHdrMetadata() {
  // Random but reasonable (in the sense of a valid range) HDR metadata.
  HdrMetadata hdr_metadata;
  hdr_metadata.mastering_metadata.luminance_max = 2000.0;
  hdr_metadata.mastering_metadata.luminance_min = 2.0001;
  hdr_metadata.mastering_metadata.primary_r.x = 0.3003;
  hdr_metadata.mastering_metadata.primary_r.y = 0.4004;
  hdr_metadata.mastering_metadata.primary_g.x = 0.3201;
  hdr_metadata.mastering_metadata.primary_g.y = 0.4604;
  hdr_metadata.mastering_metadata.primary_b.x = 0.3409;
  hdr_metadata.mastering_metadata.primary_b.y = 0.4907;
  hdr_metadata.mastering_metadata.white_point.x = 0.4103;
  hdr_metadata.mastering_metadata.white_point.y = 0.4806;
  hdr_metadata.max_content_light_level = 2345;
  hdr_metadata.max_frame_average_light_level = 1789;
  return hdr_metadata;
}

ColorSpace CreateTestColorSpace(bool with_hdr_metadata) {
  HdrMetadata hdr_metadata = CreateTestHdrMetadata();
  return ColorSpace(
      ColorSpace::PrimaryID::kBT709, ColorSpace::TransferID::kGAMMA22,
      ColorSpace::MatrixID::kSMPTE2085, ColorSpace::RangeID::kFull,
      ColorSpace::ChromaSiting::kCollocated,
      ColorSpace::ChromaSiting::kCollocated,
      with_hdr_metadata ? &hdr_metadata : nullptr);
}

RtpPacketInfos CreatePacketInfos(size_t count) {
  return RtpPacketInfos(RtpPacketInfos::vector_type(count));
}

}  // namespace webrtc
