/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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
#ifndef RTC_BASE_SOCKET_SERVER_H_
#define RTC_BASE_SOCKET_SERVER_H_

#include <memory>

#include "api/units/time_delta.h"
#include "rtc_base/event.h"
#include "rtc_base/socket_factory.h"

namespace rtc {

class Thread;
// Needs to be forward declared because there's a circular dependency between
// NetworkMonitor and Thread.
// TODO(deadbeef): Fix this.
class NetworkBinderInterface;

// Provides the ability to wait for activity on a set of sockets.  The Thread
// class provides a nice wrapper on a socket server.
//
// The server is also a socket factory.  The sockets it creates will be
// notified of asynchronous I/O from this server's Wait method.
class SocketServer : public SocketFactory {
 public:
  static constexpr webrtc::TimeDelta kForever = rtc::Event::kForever;

  static std::unique_ptr<SocketServer> CreateDefault();
  // When the socket server is installed into a Thread, this function is called
  // to allow the socket server to use the thread's message queue for any
  // messaging that it might need to perform. It is also called with a null
  // argument before the thread is destroyed.
  virtual void SetMessageQueue(Thread* /* queue */) {}

  // Sleeps until:
  //  1) `max_wait_duration` has elapsed (unless `max_wait_duration` ==
  //  `kForever`)
  // 2) WakeUp() is called
  // While sleeping, I/O is performed if process_io is true.
  virtual bool Wait(webrtc::TimeDelta max_wait_duration, bool process_io) = 0;

  // Causes the current wait (if one is in progress) to wake up.
  virtual void WakeUp() = 0;

  // A network binder will bind the created sockets to a network.
  // It is only used in PhysicalSocketServer.
  void set_network_binder(NetworkBinderInterface* binder) {
    network_binder_ = binder;
  }
  NetworkBinderInterface* network_binder() const { return network_binder_; }

 private:
  NetworkBinderInterface* network_binder_ = nullptr;
};

}  // namespace rtc

#endif  // RTC_BASE_SOCKET_SERVER_H_
