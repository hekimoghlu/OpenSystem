/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 14, 2024.
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
#import "RTCPeerConnectionFactoryOptions+Private.h"

#include "rtc_base/network_constants.h"

namespace {

void setNetworkBit(webrtc::PeerConnectionFactoryInterface::Options* options,
                   rtc::AdapterType type,
                   bool ignore) {
  if (ignore) {
    options->network_ignore_mask |= type;
  } else {
    options->network_ignore_mask &= ~type;
  }
}
}  // namespace

@implementation RTCPeerConnectionFactoryOptions

@synthesize disableEncryption = _disableEncryption;
@synthesize disableNetworkMonitor = _disableNetworkMonitor;
@synthesize ignoreLoopbackNetworkAdapter = _ignoreLoopbackNetworkAdapter;
@synthesize ignoreVPNNetworkAdapter = _ignoreVPNNetworkAdapter;
@synthesize ignoreCellularNetworkAdapter = _ignoreCellularNetworkAdapter;
@synthesize ignoreWiFiNetworkAdapter = _ignoreWiFiNetworkAdapter;
@synthesize ignoreEthernetNetworkAdapter = _ignoreEthernetNetworkAdapter;

- (instancetype)init {
  return [super init];
}

- (webrtc::PeerConnectionFactoryInterface::Options)nativeOptions {
  webrtc::PeerConnectionFactoryInterface::Options options;
  options.disable_encryption = self.disableEncryption;
  options.disable_network_monitor = self.disableNetworkMonitor;

  setNetworkBit(&options, rtc::ADAPTER_TYPE_LOOPBACK, self.ignoreLoopbackNetworkAdapter);
  setNetworkBit(&options, rtc::ADAPTER_TYPE_VPN, self.ignoreVPNNetworkAdapter);
  setNetworkBit(&options, rtc::ADAPTER_TYPE_CELLULAR, self.ignoreCellularNetworkAdapter);
  setNetworkBit(&options, rtc::ADAPTER_TYPE_WIFI, self.ignoreWiFiNetworkAdapter);
  setNetworkBit(&options, rtc::ADAPTER_TYPE_ETHERNET, self.ignoreEthernetNetworkAdapter);

  return options;
}

@end
