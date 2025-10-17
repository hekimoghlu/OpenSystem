/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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
#ifndef SDK_ANDROID_NATIVE_API_STACKTRACE_STACKTRACE_H_
#define SDK_ANDROID_NATIVE_API_STACKTRACE_STACKTRACE_H_

#include <string>
#include <vector>

namespace webrtc {

struct StackTraceElement {
  // Pathname of shared object (.so file) that contains address.
  const char* shared_object_path;
  // Execution address relative to the .so base address. This matches the
  // addresses you get with "nm", "objdump", and "ndk-stack", as long as the
  // code is compiled with position-independent code. Android requires
  // position-independent code since Lollipop.
  uint32_t relative_address;
  // Name of symbol whose definition overlaps the address. This value is null
  // when symbol names are stripped.
  const char* symbol_name;
};

// Utility to unwind stack for a given thread on Android ARM devices. This works
// on top of unwind.h and unwinds native (C++) stack traces only.
std::vector<StackTraceElement> GetStackTrace(int tid);

// Unwind the stack of the current thread.
std::vector<StackTraceElement> GetStackTrace();

// Get a string representation of the stack trace in a format ndk-stack accepts.
std::string StackTraceToString(
    const std::vector<StackTraceElement>& stack_trace);

}  // namespace webrtc

#endif  // SDK_ANDROID_NATIVE_API_STACKTRACE_STACKTRACE_H_
