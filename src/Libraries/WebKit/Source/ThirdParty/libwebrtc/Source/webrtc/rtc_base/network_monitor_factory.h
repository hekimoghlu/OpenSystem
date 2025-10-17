/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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
#ifndef RTC_BASE_NETWORK_MONITOR_FACTORY_H_
#define RTC_BASE_NETWORK_MONITOR_FACTORY_H_

namespace webrtc {
class FieldTrialsView;
}  // namespace webrtc

namespace rtc {

// Forward declaring this so it's not part of the API surface; it's only
// expected to be used by Android/iOS SDK code.
class NetworkMonitorInterface;

/*
 * NetworkMonitorFactory creates NetworkMonitors.
 * Note that CreateNetworkMonitor is expected to be called on the network
 * thread with the returned object only being used on that thread thereafter.
 */
class NetworkMonitorFactory {
 public:
  virtual NetworkMonitorInterface* CreateNetworkMonitor(
      const webrtc::FieldTrialsView& field_trials) = 0;

  virtual ~NetworkMonitorFactory();

 protected:
  NetworkMonitorFactory();
};

}  // namespace rtc

#endif  // RTC_BASE_NETWORK_MONITOR_FACTORY_H_
