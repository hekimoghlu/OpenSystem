/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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
#ifndef API_CALL_TRANSPORT_H_
#define API_CALL_TRANSPORT_H_

#include <stdint.h>

#include "api/array_view.h"

namespace webrtc {

// TODO(holmer): Look into unifying this with the PacketOptions in
// asyncpacketsocket.h.
struct PacketOptions {
  PacketOptions();
  PacketOptions(const PacketOptions&);
  ~PacketOptions();

  // Negative ids are invalid and should be interpreted
  // as packet_id not being set.
  int64_t packet_id = -1;
  // Whether this is an audio or video packet, excluding retransmissions.
  bool is_media = true;
  bool included_in_feedback = false;
  bool included_in_allocation = false;
  // Whether this packet can be part of a packet batch at lower levels.
  bool batchable = false;
  // Whether this packet is the last of a batch.
  bool last_packet_in_batch = false;
};

class Transport {
 public:
  virtual bool SendRtp(rtc::ArrayView<const uint8_t> packet,
                       const PacketOptions& options) = 0;
  virtual bool SendRtcp(rtc::ArrayView<const uint8_t> packet) = 0;

 protected:
  virtual ~Transport() {}
};

}  // namespace webrtc

#endif  // API_CALL_TRANSPORT_H_
