/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 6, 2023.
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
#include "rtc_base/server_socket_adapters.h"

#include <string>

#include "rtc_base/byte_buffer.h"

namespace rtc {

AsyncProxyServerSocket::AsyncProxyServerSocket(Socket* socket,
                                               size_t buffer_size)
    : BufferedReadAdapter(socket, buffer_size) {}

AsyncProxyServerSocket::~AsyncProxyServerSocket() = default;

AsyncSSLServerSocket::AsyncSSLServerSocket(Socket* socket)
    : BufferedReadAdapter(socket, 1024) {
  BufferInput(true);
}

void AsyncSSLServerSocket::ProcessInput(char* data, size_t* len) {
  // We only accept client hello messages.
  const ArrayView<const uint8_t> client_hello =
      AsyncSSLSocket::SslClientHello();
  if (*len < client_hello.size()) {
    return;
  }

  if (memcmp(client_hello.data(), data, client_hello.size()) != 0) {
    Close();
    SignalCloseEvent(this, 0);
    return;
  }

  *len -= client_hello.size();

  // Clients should not send more data until the handshake is completed.
  RTC_DCHECK(*len == 0);

  const ArrayView<const uint8_t> server_hello =
      AsyncSSLSocket::SslServerHello();
  // Send a server hello back to the client.
  DirectSend(server_hello.data(), server_hello.size());

  // Handshake completed for us, redirect input to our parent.
  BufferInput(false);
}

}  // namespace rtc
