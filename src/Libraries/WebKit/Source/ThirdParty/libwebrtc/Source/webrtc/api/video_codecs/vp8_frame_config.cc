/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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
#include "api/video_codecs/vp8_frame_config.h"

#include "modules/video_coding/codecs/interface/common_constants.h"
#include "rtc_base/checks.h"

namespace webrtc {

Vp8FrameConfig::Vp8FrameConfig() : Vp8FrameConfig(kNone, kNone, kNone, false) {}

Vp8FrameConfig::Vp8FrameConfig(BufferFlags last,
                               BufferFlags golden,
                               BufferFlags arf)
    : Vp8FrameConfig(last, golden, arf, false) {}

Vp8FrameConfig::Vp8FrameConfig(BufferFlags last,
                               BufferFlags golden,
                               BufferFlags arf,
                               FreezeEntropy)
    : Vp8FrameConfig(last, golden, arf, true) {}

Vp8FrameConfig::Vp8FrameConfig(BufferFlags last,
                               BufferFlags golden,
                               BufferFlags arf,
                               bool freeze_entropy)
    : drop_frame(last == BufferFlags::kNone && golden == BufferFlags::kNone &&
                 arf == BufferFlags::kNone),
      last_buffer_flags(last),
      golden_buffer_flags(golden),
      arf_buffer_flags(arf),
      encoder_layer_id(0),
      packetizer_temporal_idx(kNoTemporalIdx),
      layer_sync(false),
      freeze_entropy(freeze_entropy),
      first_reference(Vp8BufferReference::kNone),
      second_reference(Vp8BufferReference::kNone),
      retransmission_allowed(true) {}

bool Vp8FrameConfig::References(Buffer buffer) const {
  switch (buffer) {
    case Buffer::kLast:
      return (last_buffer_flags & kReference) != 0;
    case Buffer::kGolden:
      return (golden_buffer_flags & kReference) != 0;
    case Buffer::kArf:
      return (arf_buffer_flags & kReference) != 0;
    case Buffer::kCount:
      break;
  }
  RTC_DCHECK_NOTREACHED();
  return false;
}

bool Vp8FrameConfig::Updates(Buffer buffer) const {
  switch (buffer) {
    case Buffer::kLast:
      return (last_buffer_flags & kUpdate) != 0;
    case Buffer::kGolden:
      return (golden_buffer_flags & kUpdate) != 0;
    case Buffer::kArf:
      return (arf_buffer_flags & kUpdate) != 0;
    case Buffer::kCount:
      break;
  }
  RTC_DCHECK_NOTREACHED();
  return false;
}

}  // namespace webrtc
