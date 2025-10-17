/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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
#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <malloc.h>
#include <pthread.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <unwind.h>

#include "MapData.h"
#include "backtrace.h"
#include "debug_log.h"

#if defined(__LP64__)
#define PAD_PTR "016" PRIxPTR
#else
#define PAD_PTR "08" PRIxPTR
#endif

typedef struct _Unwind_Context __unwind_context;

static MapData g_map_data;
static MapEntry g_current_code_map;

static _Unwind_Reason_Code find_current_map(__unwind_context* context, void*) {
  uintptr_t ip = _Unwind_GetIP(context);

  if (ip == 0) {
    return _URC_END_OF_STACK;
  }
  auto map = g_map_data.find(ip);
  if (map != nullptr) {
    g_current_code_map = *map;
  }
  return _URC_END_OF_STACK;
}

void backtrace_startup() {
  g_map_data.ReadMaps();
  _Unwind_Backtrace(find_current_map, nullptr);
}

void backtrace_shutdown() {}

struct stack_crawl_state_t {
  uintptr_t* frames;
  size_t frame_count;
  size_t cur_frame = 0;

  stack_crawl_state_t(uintptr_t* frames, size_t frame_count)
      : frames(frames), frame_count(frame_count) {}
};

static _Unwind_Reason_Code trace_function(__unwind_context* context, void* arg) {
  stack_crawl_state_t* state = static_cast<stack_crawl_state_t*>(arg);

  uintptr_t ip = _Unwind_GetIP(context);

  // `ip` is the address of the instruction *after* the call site in
  // `context`, so we want to back up by one instruction. This is hard for
  // every architecture except arm64, so we just make sure we're *inside*
  // that instruction, not necessarily at the start of it. (If the value
  // is too low to be valid, we just leave it alone.)
  if (ip >= 4096) {
#if defined(__aarch64__)
    ip -= 4;  // Exactly.
#elif defined(__arm__) || defined(__riscv)
    ip -= 2;  // At least.
#elif defined(__i386__) || defined(__x86_64__)
    ip -= 1;  // At least.
#endif
  }

  // Do not record the frames that fall in our own shared library.
  if (g_current_code_map.start() != 0 && (ip >= g_current_code_map.start()) &&
      ip < g_current_code_map.end()) {
    return _URC_NO_REASON;
  }

  state->frames[state->cur_frame++] = ip;
  return (state->cur_frame >= state->frame_count) ? _URC_END_OF_STACK : _URC_NO_REASON;
}

size_t backtrace_get(uintptr_t* frames, size_t frame_count) {
  stack_crawl_state_t state(frames, frame_count);
  _Unwind_Backtrace(trace_function, &state);
  return state.cur_frame;
}

std::string backtrace_string(const uintptr_t* frames, size_t frame_count) {
  if (g_map_data.NumMaps() == 0) {
    g_map_data.ReadMaps();
  }

  std::string str;

  for (size_t frame_num = 0; frame_num < frame_count; frame_num++) {
    uintptr_t offset = 0;
    const char* symbol = nullptr;

    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(frames[frame_num]), &info) != 0) {
      offset = reinterpret_cast<uintptr_t>(info.dli_saddr);
      symbol = info.dli_sname;
    } else {
      info.dli_fname = nullptr;
    }

    uintptr_t rel_pc = offset;
    const MapEntry* entry = g_map_data.find(frames[frame_num], &rel_pc);

    const char* soname = (entry != nullptr) ? entry->name().c_str() : info.dli_fname;
    if (soname == nullptr) {
      soname = "<unknown>";
    }

    char offset_buf[128];
    if (entry != nullptr && entry->elf_start_offset() != 0) {
      snprintf(offset_buf, sizeof(offset_buf), " (offset 0x%" PRIxPTR ")",
               entry->elf_start_offset());
    } else {
      offset_buf[0] = '\0';
    }

    char buf[1024];
    if (symbol != nullptr) {
      char* demangled_name = abi::__cxa_demangle(symbol, nullptr, nullptr, nullptr);
      const char* name;
      if (demangled_name != nullptr) {
        name = demangled_name;
      } else {
        name = symbol;
      }
      async_safe_format_buffer(buf, sizeof(buf),
                               "          #%02zd  pc %" PAD_PTR "  %s%s (%s+%" PRIuPTR ")\n",
                               frame_num, rel_pc, soname, offset_buf, name,
                               frames[frame_num] - offset);
      free(demangled_name);
    } else {
      async_safe_format_buffer(buf, sizeof(buf), "          #%02zd  pc %" PAD_PTR "  %s%s\n",
                               frame_num, rel_pc, soname, offset_buf);
    }
    str += buf;
  }

  return str;
}

void backtrace_log(const uintptr_t* frames, size_t frame_count) {
  g_map_data.ReadMaps();
  error_log_string(backtrace_string(frames, frame_count).c_str());
}
