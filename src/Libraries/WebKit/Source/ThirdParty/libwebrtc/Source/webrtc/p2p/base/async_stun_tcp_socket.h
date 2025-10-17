/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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
#ifndef P2P_BASE_ASYNC_STUN_TCP_SOCKET_H_
#define P2P_BASE_ASYNC_STUN_TCP_SOCKET_H_

#include <stddef.h>

#include "rtc_base/async_packet_socket.h"
#include "rtc_base/async_tcp_socket.h"
#include "rtc_base/socket.h"
#include "rtc_base/socket_address.h"

namespace cricket {

class AsyncStunTCPSocket : public rtc::AsyncTCPSocketBase {
 public:
  // Binds and connects `socket` and creates AsyncTCPSocket for
  // it. Takes ownership of `socket`. Returns NULL if bind() or
  // connect() fail (`socket` is destroyed in that case).
  static AsyncStunTCPSocket* Create(rtc::Socket* socket,
                                    const rtc::SocketAddress& bind_address,
                                    const rtc::SocketAddress& remote_address);

  explicit AsyncStunTCPSocket(rtc::Socket* socket);

  AsyncStunTCPSocket(const AsyncStunTCPSocket&) = delete;
  AsyncStunTCPSocket& operator=(const AsyncStunTCPSocket&) = delete;

  int Send(const void* pv,
           size_t cb,
           const rtc::PacketOptions& options) override;
  size_t ProcessInput(rtc::ArrayView<const uint8_t> data) override;

 private:
  // This method returns the message hdr + length written in the header.
  // This method also returns the number of padding bytes needed/added to the
  // turn message. `pad_bytes` should be used only when `is_turn` is true.
  size_t GetExpectedLength(const void* data, size_t len, int* pad_bytes);
};

}  // namespace cricket

#endif  // P2P_BASE_ASYNC_STUN_TCP_SOCKET_H_
