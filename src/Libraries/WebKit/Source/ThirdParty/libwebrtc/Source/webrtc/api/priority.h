/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 24, 2023.
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
#ifndef API_PRIORITY_H_
#define API_PRIORITY_H_

#include <stdint.h>

#include "rtc_base/checks.h"
#include "rtc_base/strong_alias.h"

namespace webrtc {

// GENERATED_JAVA_ENUM_PACKAGE: org.webrtc
enum class Priority {
  kVeryLow,
  kLow,
  kMedium,
  kHigh,
};

class PriorityValue
    : public webrtc::StrongAlias<class PriorityValueTag, uint16_t> {
 public:
  explicit PriorityValue(Priority priority) {
    switch (priority) {
      case Priority::kVeryLow:
        value_ = 128;
        break;
      case Priority::kLow:
        value_ = 256;
        break;
      case Priority::kMedium:
        value_ = 512;
        break;
      case Priority::kHigh:
        value_ = 1024;
        break;
      default:
        RTC_CHECK_NOTREACHED();
    }
  }

  explicit PriorityValue(uint16_t priority) : StrongAlias(priority) {}
};

}  // namespace webrtc

#endif  // API_PRIORITY_H_
