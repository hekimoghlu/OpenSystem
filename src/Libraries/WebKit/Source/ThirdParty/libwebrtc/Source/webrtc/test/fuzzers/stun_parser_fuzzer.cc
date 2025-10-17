/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 28, 2022.
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
#include <stddef.h>
#include <stdint.h>

#include "api/transport/stun.h"

namespace webrtc {
void FuzzOneInput(const uint8_t* data, size_t size) {

  // Normally we'd check the integrity first, but those checks are
  // fuzzed separately in stun_validator_fuzzer.cc. We still want to
  // fuzz this target since the integrity checks could be forged by a
  // malicious adversary who receives a call.
  std::unique_ptr<cricket::IceMessage> stun_msg(new cricket::IceMessage());
  rtc::ByteBufferReader buf(rtc::MakeArrayView(data, size));
  stun_msg->Read(&buf);
  stun_msg->ValidateMessageIntegrity("");
}
}  // namespace webrtc
