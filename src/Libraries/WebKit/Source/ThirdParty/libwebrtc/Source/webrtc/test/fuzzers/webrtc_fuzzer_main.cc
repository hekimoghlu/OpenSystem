/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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
// This file is intended to provide a common interface for fuzzing functions.
// It's intended to set sane defaults, such as removing logging for further
// fuzzing efficiency.

#include "rtc_base/logging.h"

namespace {
bool g_initialized = false;
void InitializeWebRtcFuzzDefaults() {
  if (g_initialized)
    return;

// Remove default logging to prevent huge slowdowns.
// TODO(pbos): Disable in Chromium: http://crbug.com/561667
#if !defined(WEBRTC_CHROMIUM_BUILD)
  rtc::LogMessage::LogToDebug(rtc::LS_NONE);
#endif  // !defined(WEBRTC_CHROMIUM_BUILD)

  g_initialized = true;
}
}  // namespace

namespace webrtc {
extern void FuzzOneInput(const uint8_t* data, size_t size);
}  // namespace webrtc

extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data, size_t size) {
  InitializeWebRtcFuzzDefaults();
  webrtc::FuzzOneInput(data, size);
  return 0;
}
