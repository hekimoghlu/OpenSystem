/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 23, 2025.
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
#include "pc/ice_transport.h"

#include "api/sequence_checker.h"

namespace webrtc {

IceTransportWithPointer::~IceTransportWithPointer() {
  // We depend on the networking thread to call Clear() before dropping
  // its last reference to this object; if the destructor is called
  // on the networking thread, it's OK to not have called Clear().
  if (internal_) {
    RTC_DCHECK_RUN_ON(creator_thread_);
  }
}

cricket::IceTransportInternal* IceTransportWithPointer::internal() {
  RTC_DCHECK_RUN_ON(creator_thread_);
  return internal_;
}

void IceTransportWithPointer::Clear() {
  RTC_DCHECK_RUN_ON(creator_thread_);
  internal_ = nullptr;
}

}  // namespace webrtc
