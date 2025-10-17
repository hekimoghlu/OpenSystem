/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 30, 2024.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TIMESTAMP_SCALER_H_
#define MODULES_AUDIO_CODING_NETEQ_TIMESTAMP_SCALER_H_

#include "modules/audio_coding/neteq/packet.h"

namespace webrtc {

// Forward declaration.
class DecoderDatabase;

// This class scales timestamps for codecs that need timestamp scaling.
// This is done for codecs where one RTP timestamp does not correspond to
// one sample.
class TimestampScaler {
 public:
  explicit TimestampScaler(const DecoderDatabase& decoder_database)
      : first_packet_received_(false),
        numerator_(1),
        denominator_(1),
        external_ref_(0),
        internal_ref_(0),
        decoder_database_(decoder_database) {}

  virtual ~TimestampScaler() {}

  TimestampScaler(const TimestampScaler&) = delete;
  TimestampScaler& operator=(const TimestampScaler&) = delete;

  // Start over.
  virtual void Reset();

  // Scale the timestamp in `packet` from external to internal.
  virtual void ToInternal(Packet* packet);

  // Scale the timestamp for all packets in `packet_list` from external to
  // internal.
  virtual void ToInternal(PacketList* packet_list);

  // Returns the internal equivalent of `external_timestamp`, given the
  // RTP payload type `rtp_payload_type`.
  virtual uint32_t ToInternal(uint32_t external_timestamp,
                              uint8_t rtp_payload_type);

  // Scales back to external timestamp. This is the inverse of ToInternal().
  virtual uint32_t ToExternal(uint32_t internal_timestamp) const;

 private:
  bool first_packet_received_;
  int numerator_;
  int denominator_;
  uint32_t external_ref_;
  uint32_t internal_ref_;
  const DecoderDatabase& decoder_database_;
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_TIMESTAMP_SCALER_H_
