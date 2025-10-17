/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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
#ifndef RTC_BASE_NAT_TYPES_H_
#define RTC_BASE_NAT_TYPES_H_

namespace rtc {

/* Identifies each type of NAT that can be simulated. */
enum NATType {
  NAT_OPEN_CONE,
  NAT_ADDR_RESTRICTED,
  NAT_PORT_RESTRICTED,
  NAT_SYMMETRIC
};

// Implements the rules for each specific type of NAT.
class NAT {
 public:
  virtual ~NAT() {}

  // Determines whether this NAT uses both source and destination address when
  // checking whether a mapping already exists.
  virtual bool IsSymmetric() = 0;

  // Determines whether this NAT drops packets received from a different IP
  // the one last sent to.
  virtual bool FiltersIP() = 0;

  // Determines whether this NAT drops packets received from a different port
  // the one last sent to.
  virtual bool FiltersPort() = 0;

  // Returns an implementation of the given type of NAT.
  static NAT* Create(NATType type);
};

}  // namespace rtc

#endif  // RTC_BASE_NAT_TYPES_H_
