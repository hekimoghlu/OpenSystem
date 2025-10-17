/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 27, 2025.
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
#include "rtc_base/nat_types.h"

#include "rtc_base/checks.h"

namespace rtc {

class SymmetricNAT : public NAT {
 public:
  bool IsSymmetric() override { return true; }
  bool FiltersIP() override { return true; }
  bool FiltersPort() override { return true; }
};

class OpenConeNAT : public NAT {
 public:
  bool IsSymmetric() override { return false; }
  bool FiltersIP() override { return false; }
  bool FiltersPort() override { return false; }
};

class AddressRestrictedNAT : public NAT {
 public:
  bool IsSymmetric() override { return false; }
  bool FiltersIP() override { return true; }
  bool FiltersPort() override { return false; }
};

class PortRestrictedNAT : public NAT {
 public:
  bool IsSymmetric() override { return false; }
  bool FiltersIP() override { return true; }
  bool FiltersPort() override { return true; }
};

NAT* NAT::Create(NATType type) {
  switch (type) {
    case NAT_OPEN_CONE:
      return new OpenConeNAT();
    case NAT_ADDR_RESTRICTED:
      return new AddressRestrictedNAT();
    case NAT_PORT_RESTRICTED:
      return new PortRestrictedNAT();
    case NAT_SYMMETRIC:
      return new SymmetricNAT();
    default:
      RTC_DCHECK_NOTREACHED();
      return 0;
  }
}

}  // namespace rtc
