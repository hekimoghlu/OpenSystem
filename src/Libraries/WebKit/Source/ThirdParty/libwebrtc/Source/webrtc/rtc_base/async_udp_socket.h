/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 23, 2023.
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
#ifndef RTC_BASE_ASYNC_UDP_SOCKET_H_
#define RTC_BASE_ASYNC_UDP_SOCKET_H_

#include <stddef.h>

#include <memory>
#include <optional>

#include "api/sequence_checker.h"
#include "api/units/time_delta.h"
#include "rtc_base/async_packet_socket.h"
#include "rtc_base/buffer.h"
#include "rtc_base/socket.h"
#include "rtc_base/socket_address.h"
#include "rtc_base/socket_factory.h"
#include "rtc_base/system/no_unique_address.h"
#include "rtc_base/thread_annotations.h"

namespace rtc {

// Provides the ability to receive packets asynchronously.  Sends are not
// buffered since it is acceptable to drop packets under high load.
class AsyncUDPSocket : public AsyncPacketSocket {
 public:
  // Binds `socket` and creates AsyncUDPSocket for it. Takes ownership
  // of `socket`. Returns null if bind() fails (`socket` is destroyed
  // in that case).
  static AsyncUDPSocket* Create(Socket* socket,
                                const SocketAddress& bind_address);
  // Creates a new socket for sending asynchronous UDP packets using an
  // asynchronous socket from the given factory.
  static AsyncUDPSocket* Create(SocketFactory* factory,
                                const SocketAddress& bind_address);
  explicit AsyncUDPSocket(Socket* socket);
  ~AsyncUDPSocket() = default;

  SocketAddress GetLocalAddress() const override;
  SocketAddress GetRemoteAddress() const override;
  int Send(const void* pv,
           size_t cb,
           const rtc::PacketOptions& options) override;
  int SendTo(const void* pv,
             size_t cb,
             const SocketAddress& addr,
             const rtc::PacketOptions& options) override;
  int Close() override;

  State GetState() const override;
  int GetOption(Socket::Option opt, int* value) override;
  int SetOption(Socket::Option opt, int value) override;
  int GetError() const override;
  void SetError(int error) override;

 private:
  // Called when the underlying socket is ready to be read from.
  void OnReadEvent(Socket* socket);
  // Called when the underlying socket is ready to send.
  void OnWriteEvent(Socket* socket);

  RTC_NO_UNIQUE_ADDRESS webrtc::SequenceChecker sequence_checker_;
  std::unique_ptr<Socket> socket_;
  bool has_set_ect1_options_ = false;
  rtc::Buffer buffer_ RTC_GUARDED_BY(sequence_checker_);
  std::optional<webrtc::TimeDelta> socket_time_offset_
      RTC_GUARDED_BY(sequence_checker_);
};

}  // namespace rtc

#endif  // RTC_BASE_ASYNC_UDP_SOCKET_H_
