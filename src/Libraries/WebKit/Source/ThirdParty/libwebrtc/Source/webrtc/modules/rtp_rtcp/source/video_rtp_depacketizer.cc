/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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
#include "modules/rtp_rtcp/source/video_rtp_depacketizer.h"

#include <stddef.h>
#include <stdint.h>

#include "api/array_view.h"
#include "api/scoped_refptr.h"
#include "api/video/encoded_image.h"
#include "rtc_base/checks.h"

namespace webrtc {

rtc::scoped_refptr<EncodedImageBuffer> VideoRtpDepacketizer::AssembleFrame(
    rtc::ArrayView<const rtc::ArrayView<const uint8_t>> rtp_payloads) {
  size_t frame_size = 0;
  for (rtc::ArrayView<const uint8_t> payload : rtp_payloads) {
    frame_size += payload.size();
  }

  rtc::scoped_refptr<EncodedImageBuffer> bitstream =
      EncodedImageBuffer::Create(frame_size);

  uint8_t* write_at = bitstream->data();
  for (rtc::ArrayView<const uint8_t> payload : rtp_payloads) {
    memcpy(write_at, payload.data(), payload.size());
    write_at += payload.size();
  }
  RTC_DCHECK_EQ(write_at - bitstream->data(), bitstream->size());
  return bitstream;
}

}  // namespace webrtc
