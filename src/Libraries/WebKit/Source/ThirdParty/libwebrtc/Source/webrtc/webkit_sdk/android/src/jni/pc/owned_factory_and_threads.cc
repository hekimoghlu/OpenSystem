/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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
#include "sdk/android/src/jni/pc/owned_factory_and_threads.h"

#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

OwnedFactoryAndThreads::OwnedFactoryAndThreads(
    std::unique_ptr<rtc::SocketFactory> socket_factory,
    std::unique_ptr<rtc::Thread> network_thread,
    std::unique_ptr<rtc::Thread> worker_thread,
    std::unique_ptr<rtc::Thread> signaling_thread,
    const rtc::scoped_refptr<PeerConnectionFactoryInterface>& factory)
    : socket_factory_(std::move(socket_factory)),
      network_thread_(std::move(network_thread)),
      worker_thread_(std::move(worker_thread)),
      signaling_thread_(std::move(signaling_thread)),
      factory_(factory) {}

}  // namespace jni
}  // namespace webrtc
