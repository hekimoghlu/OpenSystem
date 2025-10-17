/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 7, 2025.
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
#include "modules/portal/pipewire_utils.h"

#include <pipewire/pipewire.h>

#include "rtc_base/sanitizer.h"

#if defined(WEBRTC_DLOPEN_PIPEWIRE)
#include "modules/portal/pipewire_stubs.h"
#endif  // defined(WEBRTC_DLOPEN_PIPEWIRE)

namespace webrtc {

RTC_NO_SANITIZE("cfi-icall")
bool InitializePipeWire() {
#if defined(WEBRTC_DLOPEN_PIPEWIRE)
  static constexpr char kPipeWireLib[] = "libpipewire-0.3.so.0";

  using modules_portal::InitializeStubs;
  using modules_portal::kModulePipewire;

  modules_portal::StubPathMap paths;

  // Check if the PipeWire library is available.
  paths[kModulePipewire].push_back(kPipeWireLib);

  static bool result = InitializeStubs(paths);

  return result;
#else
  return true;
#endif  // defined(WEBRTC_DLOPEN_PIPEWIRE)
}

PipeWireThreadLoopLock::PipeWireThreadLoopLock(pw_thread_loop* loop)
    : loop_(loop) {
  pw_thread_loop_lock(loop_);
}

PipeWireThreadLoopLock::~PipeWireThreadLoopLock() {
  pw_thread_loop_unlock(loop_);
}

}  // namespace webrtc
