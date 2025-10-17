/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 17, 2024.
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
#include <cxxabi.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdint.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <android-base/stringprintf.h>
#include <unwindstack/AndroidUnwinder.h>
#include <unwindstack/Unwinder.h>

#include "UnwindBacktrace.h"
#include "debug_log.h"

#if defined(__LP64__)
#define PAD_PTR "016" PRIx64
#else
#define PAD_PTR "08" PRIx64
#endif

bool Unwind(std::vector<uintptr_t>* frames, std::vector<unwindstack::FrameData>* frame_info,
            size_t max_frames) {
  [[language::Core::no_destroy]] static unwindstack::AndroidLocalUnwinder unwinder(
      std::vector<std::string>{"libc_malloc_debug.so"});
  unwindstack::AndroidUnwinderData data(max_frames);
  if (!unwinder.Unwind(data)) {
    frames->clear();
    frame_info->clear();
    return false;
  }

  frames->resize(data.frames.size());
  for (const auto& frame : data.frames) {
    frames->at(frame.num) = frame.pc;
  }
  *frame_info = std::move(data.frames);
  return true;
}

void UnwindLog(const std::vector<unwindstack::FrameData>& frame_info) {
  for (size_t i = 0; i < frame_info.size(); i++) {
    const unwindstack::FrameData* info = &frame_info[i];
    auto map_info = info->map_info;

    std::string line = android::base::StringPrintf("          #%0zd  pc %" PAD_PTR "  ", i, info->rel_pc);
    if (map_info != nullptr && map_info->offset() != 0) {
      line += android::base::StringPrintf("(offset 0x%" PRIx64 ") ", map_info->offset());
    }

    if (map_info == nullptr) {
      line += "<unknown>";
    } else if (map_info->name().empty()) {
      line += android::base::StringPrintf("<anonymous:%" PRIx64 ">", map_info->start());
    } else {
      line += map_info->name();
    }

    if (!info->function_name.empty()) {
      line += " (";
      char* demangled_name =
          abi::__cxa_demangle(info->function_name.c_str(), nullptr, nullptr, nullptr);
      if (demangled_name != nullptr) {
        line += demangled_name;
        free(demangled_name);
      } else {
        line += info->function_name;
      }
      if (info->function_offset != 0) {
        line += "+" + std::to_string(info->function_offset);
      }
      line += ")";
    }
    error_log_string(line.c_str());
  }
}
