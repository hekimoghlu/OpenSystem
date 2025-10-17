/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 21, 2024.
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
#ifndef RTC_BASE_IFADDRS_CONVERTER_H_
#define RTC_BASE_IFADDRS_CONVERTER_H_

#if defined(WEBRTC_ANDROID)
#include "rtc_base/ifaddrs_android.h"
#else
#include <ifaddrs.h>
#endif  // WEBRTC_ANDROID

#include "rtc_base/ip_address.h"

namespace rtc {

// This class converts native interface addresses to our internal IPAddress
// class. Subclasses should override ConvertNativeToIPAttributes to implement
// the different ways of retrieving IPv6 attributes for various POSIX platforms.
class IfAddrsConverter {
 public:
  IfAddrsConverter();
  virtual ~IfAddrsConverter();
  virtual bool ConvertIfAddrsToIPAddress(const struct ifaddrs* interface,
                                         InterfaceAddress* ipaddress,
                                         IPAddress* mask);

 protected:
  virtual bool ConvertNativeAttributesToIPAttributes(
      const struct ifaddrs* interface,
      int* ip_attributes);
};

IfAddrsConverter* CreateIfAddrsConverter();

}  // namespace rtc

#endif  // RTC_BASE_IFADDRS_CONVERTER_H_
