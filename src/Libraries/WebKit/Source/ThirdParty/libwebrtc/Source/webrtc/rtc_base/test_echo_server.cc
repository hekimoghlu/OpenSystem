/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 9, 2024.
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
#include "rtc_base/test_echo_server.h"

#include "rtc_base/socket_server.h"

namespace rtc {

TestEchoServer::TestEchoServer(Thread* thread, const SocketAddress& addr)
    : server_socket_(
          thread->socketserver()->CreateSocket(addr.family(), SOCK_STREAM)) {
  server_socket_->Bind(addr);
  server_socket_->Listen(5);
  server_socket_->SignalReadEvent.connect(this, &TestEchoServer::OnAccept);
}

TestEchoServer::~TestEchoServer() {
  for (ClientList::iterator it = client_sockets_.begin();
       it != client_sockets_.end(); ++it) {
    delete *it;
  }
}

}  // namespace rtc
