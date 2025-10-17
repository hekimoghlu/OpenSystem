/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
#include "video/transport_adapter.h"

#include "rtc_base/checks.h"

namespace webrtc {
namespace internal {

TransportAdapter::TransportAdapter(Transport* transport)
    : transport_(transport), enabled_(false) {
  RTC_DCHECK(nullptr != transport);
}

TransportAdapter::~TransportAdapter() = default;

bool TransportAdapter::SendRtp(rtc::ArrayView<const uint8_t> packet,
                               const PacketOptions& options) {
  if (!enabled_.load())
    return false;

  return transport_->SendRtp(packet, options);
}

bool TransportAdapter::SendRtcp(rtc::ArrayView<const uint8_t> packet) {
  if (!enabled_.load())
    return false;

  return transport_->SendRtcp(packet);
}

void TransportAdapter::Enable() {
  enabled_.store(true);
}

void TransportAdapter::Disable() {
  enabled_.store(false);
}

}  // namespace internal
}  // namespace webrtc
