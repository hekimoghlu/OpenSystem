/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 13, 2022.
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
#ifndef SDK_OBJC_NATIVE_SRC_NETWORK_MONITOR_OBSERVER_H_
#define SDK_OBJC_NATIVE_SRC_NETWORK_MONITOR_OBSERVER_H_

#include <map>
#include <string>

#include "rtc_base/network_constants.h"
#include "rtc_base/thread.h"

namespace webrtc {

// Observer interface for listening to NWPathMonitor updates.
class NetworkMonitorObserver {
 public:
  // Called when a path update occurs, on network monitor dispatch queue.
  //
  // |adapter_type_by_name| is a map from interface name (i.e. "pdp_ip0") to
  // adapter type, for all available interfaces on the current path. If an
  // interface name isn't present it can be assumed to be unavailable.
  virtual void OnPathUpdate(
      std::map<std::string, rtc::AdapterType> adapter_type_by_name) = 0;

 protected:
  virtual ~NetworkMonitorObserver() {}
};

}  // namespace webrtc

#endif  //  SDK_OBJC_NATIVE_SRC_AUDIO_AUDIO_SESSION_OBSERVER_H_
