/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 22, 2024.
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
#ifndef RTC_BASE_SOCKET_ADAPTERS_H_
#define RTC_BASE_SOCKET_ADAPTERS_H_

#include <string>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "rtc_base/async_socket.h"

namespace rtc {

struct HttpAuthContext;
class ByteBufferReader;
class ByteBufferWriter;

///////////////////////////////////////////////////////////////////////////////

// Implements a socket adapter that can buffer and process data internally,
// as in the case of connecting to a proxy, where you must speak the proxy
// protocol before commencing normal socket behavior.
class BufferedReadAdapter : public AsyncSocketAdapter {
 public:
  BufferedReadAdapter(Socket* socket, size_t buffer_size);
  ~BufferedReadAdapter() override;

  BufferedReadAdapter(const BufferedReadAdapter&) = delete;
  BufferedReadAdapter& operator=(const BufferedReadAdapter&) = delete;

  int Send(const void* pv, size_t cb) override;
  int Recv(void* pv, size_t cb, int64_t* timestamp) override;

 protected:
  int DirectSend(const void* pv, size_t cb) {
    return AsyncSocketAdapter::Send(pv, cb);
  }

  void BufferInput(bool on = true);
  virtual void ProcessInput(char* data, size_t* len) = 0;

  void OnReadEvent(Socket* socket) override;

 private:
  char* buffer_;
  size_t buffer_size_, data_len_;
  bool buffering_;
};

///////////////////////////////////////////////////////////////////////////////

// Implements a socket adapter that performs the client side of a
// fake SSL handshake. Used for "ssltcp" P2P functionality.
class AsyncSSLSocket : public BufferedReadAdapter {
 public:
  static ArrayView<const uint8_t> SslClientHello();
  static ArrayView<const uint8_t> SslServerHello();

  explicit AsyncSSLSocket(Socket* socket);

  AsyncSSLSocket(const AsyncSSLSocket&) = delete;
  AsyncSSLSocket& operator=(const AsyncSSLSocket&) = delete;

  int Connect(const SocketAddress& addr) override;

 protected:
  void OnConnectEvent(Socket* socket) override;
  void ProcessInput(char* data, size_t* len) override;
};

}  // namespace rtc

#endif  // RTC_BASE_SOCKET_ADAPTERS_H_
