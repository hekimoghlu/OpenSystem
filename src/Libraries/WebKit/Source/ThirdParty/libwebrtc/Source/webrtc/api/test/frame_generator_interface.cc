/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
#include "api/test/frame_generator_interface.h"

#include "rtc_base/checks.h"

namespace webrtc {
namespace test {

// static
const char* FrameGeneratorInterface::OutputTypeToString(
    FrameGeneratorInterface::OutputType type) {
  switch (type) {
    case OutputType::kI420:
      return "I420";
    case OutputType::kI420A:
      return "I420A";
    case OutputType::kI010:
      return "I010";
    case OutputType::kNV12:
      return "NV12";
    default:
      RTC_DCHECK_NOTREACHED();
  }
}

}  // namespace test
}  // namespace webrtc
