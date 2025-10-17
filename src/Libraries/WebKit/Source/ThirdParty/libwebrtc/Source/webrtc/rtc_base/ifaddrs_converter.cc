/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 24, 2021.
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
#include "rtc_base/ifaddrs_converter.h"

namespace rtc {

IfAddrsConverter::IfAddrsConverter() {}

IfAddrsConverter::~IfAddrsConverter() {}

bool IfAddrsConverter::ConvertIfAddrsToIPAddress(
    const struct ifaddrs* interface,
    InterfaceAddress* ip,
    IPAddress* mask) {
  switch (interface->ifa_addr->sa_family) {
    case AF_INET: {
      *ip = InterfaceAddress(IPAddress(
          reinterpret_cast<sockaddr_in*>(interface->ifa_addr)->sin_addr));
      *mask = IPAddress(
          reinterpret_cast<sockaddr_in*>(interface->ifa_netmask)->sin_addr);
      return true;
    }
    case AF_INET6: {
      int ip_attributes = IPV6_ADDRESS_FLAG_NONE;
      if (!ConvertNativeAttributesToIPAttributes(interface, &ip_attributes)) {
        return false;
      }
      *ip = InterfaceAddress(
          reinterpret_cast<sockaddr_in6*>(interface->ifa_addr)->sin6_addr,
          ip_attributes);
      *mask = IPAddress(
          reinterpret_cast<sockaddr_in6*>(interface->ifa_netmask)->sin6_addr);
      return true;
    }
    default: {
      return false;
    }
  }
}

bool IfAddrsConverter::ConvertNativeAttributesToIPAttributes(
    const struct ifaddrs* /* interface */,
    int* ip_attributes) {
  *ip_attributes = IPV6_ADDRESS_FLAG_NONE;
  return true;
}

#if !defined(WEBRTC_MAC)
// For MAC and IOS, it's defined in macifaddrs_converter.cc
IfAddrsConverter* CreateIfAddrsConverter() {
  return new IfAddrsConverter();
}
#endif
}  // namespace rtc
