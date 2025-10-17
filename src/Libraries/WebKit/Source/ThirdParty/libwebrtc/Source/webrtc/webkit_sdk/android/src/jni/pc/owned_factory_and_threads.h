/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 1, 2023.
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
#ifndef SDK_ANDROID_SRC_JNI_PC_OWNED_FACTORY_AND_THREADS_H_
#define SDK_ANDROID_SRC_JNI_PC_OWNED_FACTORY_AND_THREADS_H_

#include <jni.h>

#include <memory>
#include <utility>

#include "api/peer_connection_interface.h"
#include "rtc_base/thread.h"

namespace webrtc {
namespace jni {

// Helper struct for working around the fact that CreatePeerConnectionFactory()
// comes in two flavors: either entirely automagical (constructing its own
// threads and deleting them on teardown, but no external codec factory support)
// or entirely manual (requires caller to delete threads after factory
// teardown).  This struct takes ownership of its ctor's arguments to present a
// single thing for Java to hold and eventually free.
class OwnedFactoryAndThreads {
 public:
  OwnedFactoryAndThreads(
      std::unique_ptr<rtc::SocketFactory> socket_factory,
      std::unique_ptr<rtc::Thread> network_thread,
      std::unique_ptr<rtc::Thread> worker_thread,
      std::unique_ptr<rtc::Thread> signaling_thread,
      const rtc::scoped_refptr<PeerConnectionFactoryInterface>& factory);

  ~OwnedFactoryAndThreads() = default;

  PeerConnectionFactoryInterface* factory() { return factory_.get(); }
  rtc::SocketFactory* socket_factory() { return socket_factory_.get(); }
  rtc::Thread* network_thread() { return network_thread_.get(); }
  rtc::Thread* signaling_thread() { return signaling_thread_.get(); }
  rtc::Thread* worker_thread() { return worker_thread_.get(); }

 private:
  // Usually implemented by the SocketServer associated with the network thread,
  // so needs to outlive the network thread.
  const std::unique_ptr<rtc::SocketFactory> socket_factory_;
  const std::unique_ptr<rtc::Thread> network_thread_;
  const std::unique_ptr<rtc::Thread> worker_thread_;
  const std::unique_ptr<rtc::Thread> signaling_thread_;
  const rtc::scoped_refptr<PeerConnectionFactoryInterface> factory_;
};

}  // namespace jni
}  // namespace webrtc

#endif  // SDK_ANDROID_SRC_JNI_PC_OWNED_FACTORY_AND_THREADS_H_
