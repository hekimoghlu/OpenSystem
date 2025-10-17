/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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
#include "modules/audio_coding/neteq/timestamp_scaler.h"

#include "api/audio_codecs/audio_format.h"
#include "modules/audio_coding/neteq/decoder_database.h"
#include "rtc_base/checks.h"

namespace webrtc {

void TimestampScaler::Reset() {
  first_packet_received_ = false;
}

void TimestampScaler::ToInternal(Packet* packet) {
  if (!packet) {
    return;
  }
  packet->timestamp = ToInternal(packet->timestamp, packet->payload_type);
}

void TimestampScaler::ToInternal(PacketList* packet_list) {
  PacketList::iterator it;
  for (it = packet_list->begin(); it != packet_list->end(); ++it) {
    ToInternal(&(*it));
  }
}

uint32_t TimestampScaler::ToInternal(uint32_t external_timestamp,
                                     uint8_t rtp_payload_type) {
  const DecoderDatabase::DecoderInfo* info =
      decoder_database_.GetDecoderInfo(rtp_payload_type);
  if (!info) {
    // Payload type is unknown. Do not scale.
    return external_timestamp;
  }
  if (!(info->IsComfortNoise() || info->IsDtmf())) {
    // Do not change the timestamp scaling settings for DTMF or CNG.
    numerator_ = info->SampleRateHz();
    if (info->GetFormat().clockrate_hz == 0) {
      // If the clockrate is invalid (i.e. with an old-style external codec)
      // we cannot do any timestamp scaling.
      denominator_ = numerator_;
    } else {
      denominator_ = info->GetFormat().clockrate_hz;
    }
  }
  if (numerator_ != denominator_) {
    // We have a scale factor != 1.
    if (!first_packet_received_) {
      external_ref_ = external_timestamp;
      internal_ref_ = external_timestamp;
      first_packet_received_ = true;
    }
    const int64_t external_diff = int64_t{external_timestamp} - external_ref_;
    RTC_DCHECK_GT(denominator_, 0);
    external_ref_ = external_timestamp;
    internal_ref_ += (external_diff * numerator_) / denominator_;
    return internal_ref_;
  } else {
    // No scaling.
    return external_timestamp;
  }
}

uint32_t TimestampScaler::ToExternal(uint32_t internal_timestamp) const {
  if (!first_packet_received_ || (numerator_ == denominator_)) {
    // Not initialized, or scale factor is 1.
    return internal_timestamp;
  } else {
    const int64_t internal_diff = int64_t{internal_timestamp} - internal_ref_;
    RTC_DCHECK_GT(numerator_, 0);
    // Do not update references in this method.
    // Switch `denominator_` and `numerator_` to convert the other way.
    return external_ref_ + (internal_diff * denominator_) / numerator_;
  }
}

}  // namespace webrtc
