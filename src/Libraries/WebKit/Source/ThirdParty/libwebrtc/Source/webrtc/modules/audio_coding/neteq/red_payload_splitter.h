/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_RED_PAYLOAD_SPLITTER_H_
#define MODULES_AUDIO_CODING_NETEQ_RED_PAYLOAD_SPLITTER_H_

#include "modules/audio_coding/neteq/packet.h"

namespace webrtc {

class DecoderDatabase;

static const size_t kRedHeaderLength = 4;  // 4 bytes RED header.
static const size_t kRedLastHeaderLength =
    1;  // reduced size for last RED header.
// This class handles splitting of RED payloads into smaller parts.
// Codec-specific packet splitting can be performed by
// AudioDecoder::ParsePayload.
class RedPayloadSplitter {
 public:
  RedPayloadSplitter() {}

  virtual ~RedPayloadSplitter() {}

  RedPayloadSplitter(const RedPayloadSplitter&) = delete;
  RedPayloadSplitter& operator=(const RedPayloadSplitter&) = delete;

  // Splits each packet in `packet_list` into its separate RED payloads. Each
  // RED payload is packetized into a Packet. The original elements in
  // `packet_list` are properly deleted, and replaced by the new packets.
  // Note that all packets in `packet_list` must be RED payloads, i.e., have
  // RED headers according to RFC 2198 at the very beginning of the payload.
  // Returns kOK or an error.
  virtual bool SplitRed(PacketList* packet_list);

  // Checks all packets in `packet_list`. Packets that are DTMF events or
  // comfort noise payloads are kept. Except that, only one single payload type
  // is accepted. Any packet with another payload type is discarded.
  virtual void CheckRedPayloads(PacketList* packet_list,
                                const DecoderDatabase& decoder_database);
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_RED_PAYLOAD_SPLITTER_H_
