/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 7, 2023.
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
#include "api/units/data_rate.h"

#include <string>

#include "api/array_view.h"
#include "rtc_base/strings/string_builder.h"

namespace webrtc {

std::string ToString(DataRate value) {
  char buf[64];
  rtc::SimpleStringBuilder sb(buf);
  if (value.IsPlusInfinity()) {
    sb << "+inf bps";
  } else if (value.IsMinusInfinity()) {
    sb << "-inf bps";
  } else {
    if (value.bps() == 0 || value.bps() % 1000 != 0) {
      sb << value.bps() << " bps";
    } else {
      sb << value.kbps() << " kbps";
    }
  }
  return sb.str();
}
}  // namespace webrtc
