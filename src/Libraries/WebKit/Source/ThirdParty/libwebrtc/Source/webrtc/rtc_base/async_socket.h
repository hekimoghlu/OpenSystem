/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 16, 2023.
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
#ifndef RTC_BASE_ASYNC_SOCKET_H_
#define RTC_BASE_ASYNC_SOCKET_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>

#include "rtc_base/socket.h"
#include "rtc_base/socket_address.h"
#include "rtc_base/third_party/sigslot/sigslot.h"

namespace rtc {

class AsyncSocketAdapter : public Socket, public sigslot::has_slots<> {
 public:
  // Takes ownership of the passed in socket.
  // TODO(bugs.webrtc.org/6424): Change to unique_ptr here and in callers.
  explicit AsyncSocketAdapter(Socket* socket);

  SocketAddress GetLocalAddress() const override;
  SocketAddress GetRemoteAddress() const override;
  int Bind(const SocketAddress& addr) override;
  int Connect(const SocketAddress& addr) override;
  int Send(const void* pv, size_t cb) override;
  int SendTo(const void* pv, size_t cb, const SocketAddress& addr) override;
  int Recv(void* pv, size_t cb, int64_t* timestamp) override;
  int RecvFrom(void* pv,
               size_t cb,
               SocketAddress* paddr,
               int64_t* timestamp) override;
  int Listen(int backlog) override;
  Socket* Accept(SocketAddress* paddr) override;
  int Close() override;
  int GetError() const override;
  void SetError(int error) override;
  ConnState GetState() const override;
  int GetOption(Option opt, int* value) override;
  int SetOption(Option opt, int value) override;

 protected:
  virtual void OnConnectEvent(Socket* socket);
  virtual void OnReadEvent(Socket* socket);
  virtual void OnWriteEvent(Socket* socket);
  virtual void OnCloseEvent(Socket* socket, int err);

  Socket* GetSocket() const { return socket_.get(); }

 private:
  const std::unique_ptr<Socket> socket_;
};

}  // namespace rtc

#endif  // RTC_BASE_ASYNC_SOCKET_H_
