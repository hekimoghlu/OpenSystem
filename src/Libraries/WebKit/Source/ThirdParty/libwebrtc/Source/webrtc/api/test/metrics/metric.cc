/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 2, 2025.
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
#include "api/test/metrics/metric.h"

#include <string>

namespace webrtc {
namespace test {

absl::string_view ToString(Unit unit) {
  switch (unit) {
    case Unit::kMilliseconds:
      return "Milliseconds";
    case Unit::kPercent:
      return "Percent";
    case Unit::kBytes:
      return "Bytes";
    case Unit::kKilobitsPerSecond:
      return "KilobitsPerSecond";
    case Unit::kHertz:
      return "Hertz";
    case Unit::kUnitless:
      return "Unitless";
    case Unit::kCount:
      return "Count";
  }
}

absl::string_view ToString(ImprovementDirection direction) {
  switch (direction) {
    case ImprovementDirection::kBiggerIsBetter:
      return "BiggerIsBetter";
    case ImprovementDirection::kNeitherIsBetter:
      return "NeitherIsBetter";
    case ImprovementDirection::kSmallerIsBetter:
      return "SmallerIsBetter";
  }
}

}  // namespace test
}  // namespace webrtc
