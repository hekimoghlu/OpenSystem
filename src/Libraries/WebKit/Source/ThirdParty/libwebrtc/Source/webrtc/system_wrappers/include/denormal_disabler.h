/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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
#ifndef SYSTEM_WRAPPERS_INCLUDE_DENORMAL_DISABLER_H_
#define SYSTEM_WRAPPERS_INCLUDE_DENORMAL_DISABLER_H_

#include "rtc_base/system/arch.h"

namespace webrtc {

// Activates the hardware (HW) way to flush denormals (see [1]) to zero as they
// can very seriously impact performance. At destruction time restores the
// denormals handling state read by the ctor; hence, supports nested calls.
// Equals a no-op if the architecture is not x86 or ARM or if the compiler is
// not CLANG.
// [1] https://en.wikipedia.org/wiki/Denormal_number
//
// Example usage:
//
// void Foo() {
//   DenormalDisabler d;
//   ...
// }
class DenormalDisabler {
 public:
  // Ctor. If architecture and compiler are supported, stores the HW settings
  // for denormals, disables denormals and sets `disabling_activated_` to true.
  // Otherwise, only sets `disabling_activated_` to false.
  DenormalDisabler();
  // Ctor. Same as above, but also requires `enabled` to be true to disable
  // denormals.
  explicit DenormalDisabler(bool enabled);
  DenormalDisabler(const DenormalDisabler&) = delete;
  DenormalDisabler& operator=(const DenormalDisabler&) = delete;
  // Dtor. If `disabling_activated_` is true, restores the denormals HW settings
  // read by the ctor before denormals were disabled. Otherwise it's a no-op.
  ~DenormalDisabler();

  // Returns true if architecture and compiler are supported.
  static bool IsSupported();

 private:
  const int status_word_;
  const bool disabling_activated_;
};

}  // namespace webrtc

#endif  // SYSTEM_WRAPPERS_INCLUDE_DENORMAL_DISABLER_H_
