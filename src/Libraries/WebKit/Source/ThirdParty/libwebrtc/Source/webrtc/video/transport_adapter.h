/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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
#ifndef VIDEO_TRANSPORT_ADAPTER_H_
#define VIDEO_TRANSPORT_ADAPTER_H_

#include <stddef.h>
#include <stdint.h>

#include <atomic>

#include "api/call/transport.h"

namespace webrtc {
namespace internal {

class TransportAdapter : public Transport {
 public:
  explicit TransportAdapter(Transport* transport);
  ~TransportAdapter() override;

  bool SendRtp(rtc::ArrayView<const uint8_t> packet,
               const PacketOptions& options) override;
  bool SendRtcp(rtc::ArrayView<const uint8_t> packet) override;

  void Enable();
  void Disable();

 private:
  Transport* transport_;
  std::atomic<bool> enabled_;
};
}  // namespace internal
}  // namespace webrtc

#endif  // VIDEO_TRANSPORT_ADAPTER_H_
