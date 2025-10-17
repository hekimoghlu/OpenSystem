/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 20, 2021.
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
#include "rtc_base/null_socket_server.h"

#include "api/units/time_delta.h"
#include "rtc_base/checks.h"
#include "rtc_base/event.h"
#include "rtc_base/socket_server.h"

namespace rtc {

NullSocketServer::NullSocketServer() = default;
NullSocketServer::~NullSocketServer() {}

bool NullSocketServer::Wait(webrtc::TimeDelta max_wait_duration,
                            bool /* process_io */) {
  // Wait with the given timeout. Do not log a warning if we end up waiting for
  // a long time; that just means no one has any work for us, which is perfectly
  // legitimate.
  event_.Wait(max_wait_duration, /*warn_after=*/Event::kForever);
  return true;
}

void NullSocketServer::WakeUp() {
  event_.Set();
}

rtc::Socket* NullSocketServer::CreateSocket(int /* family */, int /* type */) {
  RTC_DCHECK_NOTREACHED();
  return nullptr;
}

}  // namespace rtc
