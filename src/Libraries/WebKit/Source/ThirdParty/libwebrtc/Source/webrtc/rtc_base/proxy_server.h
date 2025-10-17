/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 26, 2023.
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
#ifndef RTC_BASE_PROXY_SERVER_H_
#define RTC_BASE_PROXY_SERVER_H_

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "rtc_base/memory/fifo_buffer.h"
#include "rtc_base/server_socket_adapters.h"
#include "rtc_base/socket.h"
#include "rtc_base/socket_address.h"

namespace rtc {

class SocketFactory;

// ProxyServer is a base class that allows for easy construction of proxy
// servers. With its helper class ProxyBinding, it contains all the necessary
// logic for receiving and bridging connections. The specific client-server
// proxy protocol is implemented by an instance of the AsyncProxyServerSocket
// class; children of ProxyServer implement WrapSocket appropriately to return
// the correct protocol handler.

class ProxyBinding : public sigslot::has_slots<> {
 public:
  ProxyBinding(AsyncProxyServerSocket* in_socket, Socket* out_socket);
  ~ProxyBinding() override;

  ProxyBinding(const ProxyBinding&) = delete;
  ProxyBinding& operator=(const ProxyBinding&) = delete;

  sigslot::signal1<ProxyBinding*> SignalDestroyed;

 private:
  void OnConnectRequest(AsyncProxyServerSocket* socket,
                        const SocketAddress& addr);
  void OnInternalRead(Socket* socket);
  void OnInternalWrite(Socket* socket);
  void OnInternalClose(Socket* socket, int err);
  void OnExternalConnect(Socket* socket);
  void OnExternalRead(Socket* socket);
  void OnExternalWrite(Socket* socket);
  void OnExternalClose(Socket* socket, int err);

  static void Read(Socket* socket, FifoBuffer* buffer);
  static void Write(Socket* socket, FifoBuffer* buffer);
  void Destroy();

  static const int kBufferSize = 4096;
  std::unique_ptr<AsyncProxyServerSocket> int_socket_;
  std::unique_ptr<Socket> ext_socket_;
  bool connected_;
  FifoBuffer out_buffer_;
  FifoBuffer in_buffer_;
};

class ProxyServer : public sigslot::has_slots<> {
 public:
  ProxyServer(SocketFactory* int_factory,
              const SocketAddress& int_addr,
              SocketFactory* ext_factory,
              const SocketAddress& ext_ip);
  ~ProxyServer() override;

  ProxyServer(const ProxyServer&) = delete;
  ProxyServer& operator=(const ProxyServer&) = delete;

  // Returns the address to which the proxy server is bound
  SocketAddress GetServerAddress();

 protected:
  void OnAcceptEvent(Socket* socket);
  virtual AsyncProxyServerSocket* WrapSocket(Socket* socket) = 0;

 private:
  SocketFactory* ext_factory_;
  SocketAddress ext_ip_;
  std::unique_ptr<Socket> server_socket_;
  std::vector<std::unique_ptr<ProxyBinding>> bindings_;
};

}  // namespace rtc

#endif  // RTC_BASE_PROXY_SERVER_H_
