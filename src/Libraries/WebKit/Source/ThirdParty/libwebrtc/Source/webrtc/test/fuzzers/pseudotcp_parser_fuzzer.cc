/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 4, 2022.
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

#include "p2p/base/pseudo_tcp.h"
#include "rtc_base/thread.h"

namespace webrtc {
class FakeIPseudoTcpNotify : public cricket::IPseudoTcpNotify {
 public:
  void OnTcpOpen(cricket::PseudoTcp* tcp) {}
  void OnTcpReadable(cricket::PseudoTcp* tcp) {}
  void OnTcpWriteable(cricket::PseudoTcp* tcp) {}
  void OnTcpClosed(cricket::PseudoTcp* tcp, uint32_t error) {}

  cricket::IPseudoTcpNotify::WriteResult TcpWritePacket(cricket::PseudoTcp* tcp,
                                                        const char* buffer,
                                                        size_t len) {
    return cricket::IPseudoTcpNotify::WriteResult::WR_SUCCESS;
  }
};

struct Environment {
  explicit Environment(cricket::IPseudoTcpNotify* notifier)
      : ptcp(notifier, 0) {}

  // We need the thread to avoid some uninteresting crashes, since the
  // production code expects there to be a thread object available.
  rtc::AutoThread thread;
  cricket::PseudoTcp ptcp;
};

Environment* env = new Environment(new FakeIPseudoTcpNotify());

void FuzzOneInput(const uint8_t* data, size_t size) {
  env->ptcp.NotifyPacket(reinterpret_cast<const char*>(data), size);
}
}  // namespace webrtc
