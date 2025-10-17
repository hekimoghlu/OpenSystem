/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 13, 2023.
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
#ifndef SDK_OBJC_NATIVE_SRC_OBJC_NETWORK_MONITOR_H_
#define SDK_OBJC_NATIVE_SRC_OBJC_NETWORK_MONITOR_H_

#include <vector>

#include "webkit_sdk/objc/components/network/RTCNetworkMonitor+Private.h"
#include "webkit_sdk/objc/native/src/network_monitor_observer.h"

#include "rtc_base/async_invoker.h"
#include "rtc_base/network_monitor.h"
#include "rtc_base/network_monitor_factory.h"
#include "rtc_base/synchronization/sequence_checker.h"
#include "rtc_base/thread.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

class ObjCNetworkMonitorFactory : public rtc::NetworkMonitorFactory {
 public:
  ObjCNetworkMonitorFactory() = default;
  ~ObjCNetworkMonitorFactory() override = default;

  rtc::NetworkMonitorInterface* CreateNetworkMonitor() override;
};

class ObjCNetworkMonitor : public rtc::NetworkMonitorInterface,
                           public NetworkMonitorObserver {
 public:
  ObjCNetworkMonitor() = default;
  ~ObjCNetworkMonitor() override;

  void Start() override;
  void Stop() override;

  rtc::AdapterType GetAdapterType(const std::string& interface_name) override;
  rtc::AdapterType GetVpnUnderlyingAdapterType(
      const std::string& interface_name) override;
  rtc::NetworkPreference GetNetworkPreference(
      const std::string& interface_name) override;
  bool IsAdapterAvailable(const std::string& interface_name) override;

  // NetworkMonitorObserver override.
  // Fans out updates to observers on the correct thread.
  void OnPathUpdate(
      std::map<std::string, rtc::AdapterType> adapter_type_by_name) override;

 private:
  rtc::Thread* thread_ = nullptr;
  bool started_ = false;
  std::map<std::string, rtc::AdapterType> adapter_type_by_name_
      RTC_GUARDED_BY(thread_);
  rtc::AsyncInvoker invoker_;
  RTCNetworkMonitor* network_monitor_ = nil;
};

}  // namespace webrtc

#endif  // SDK_OBJC_NATIVE_SRC_OBJC_NETWORK_MONITOR_H_
