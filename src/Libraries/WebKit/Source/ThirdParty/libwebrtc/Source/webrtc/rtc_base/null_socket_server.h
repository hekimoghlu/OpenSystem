/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 12, 2023.
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
#ifndef RTC_BASE_NULL_SOCKET_SERVER_H_
#define RTC_BASE_NULL_SOCKET_SERVER_H_

#include "rtc_base/event.h"
#include "rtc_base/socket.h"
#include "rtc_base/socket_server.h"
#include "rtc_base/system/rtc_export.h"

namespace rtc {

class RTC_EXPORT NullSocketServer : public SocketServer {
 public:
  NullSocketServer();
  ~NullSocketServer() override;

  bool Wait(webrtc::TimeDelta max_wait_duration, bool process_io) override;
  void WakeUp() override;

  Socket* CreateSocket(int family, int type) override;

 private:
  Event event_;
};

}  // namespace rtc

#endif  // RTC_BASE_NULL_SOCKET_SERVER_H_
