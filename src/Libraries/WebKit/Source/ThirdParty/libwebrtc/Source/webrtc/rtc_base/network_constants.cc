/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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
#include "rtc_base/network_constants.h"

#include "rtc_base/checks.h"

namespace rtc {

std::string AdapterTypeToString(AdapterType type) {
  switch (type) {
    case ADAPTER_TYPE_ANY:
      return "Wildcard";
    case ADAPTER_TYPE_UNKNOWN:
      return "Unknown";
    case ADAPTER_TYPE_ETHERNET:
      return "Ethernet";
    case ADAPTER_TYPE_WIFI:
      return "Wifi";
    case ADAPTER_TYPE_CELLULAR:
      return "Cellular";
    case ADAPTER_TYPE_CELLULAR_2G:
      return "Cellular2G";
    case ADAPTER_TYPE_CELLULAR_3G:
      return "Cellular3G";
    case ADAPTER_TYPE_CELLULAR_4G:
      return "Cellular4G";
    case ADAPTER_TYPE_CELLULAR_5G:
      return "Cellular5G";
    case ADAPTER_TYPE_VPN:
      return "VPN";
    case ADAPTER_TYPE_LOOPBACK:
      return "Loopback";
    default:
      RTC_DCHECK_NOTREACHED() << "Invalid type " << type;
      return std::string();
  }
}

}  // namespace rtc
