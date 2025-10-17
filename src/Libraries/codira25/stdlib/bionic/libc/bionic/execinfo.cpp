/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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
#include <dlfcn.h>
#include <execinfo.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <unwind.h>

#include "private/ScopedFd.h"

struct StackState {
  void** frames;
  int frame_count;
  int cur_frame = 0;

  StackState(void** frames, int frame_count) : frames(frames), frame_count(frame_count) {}
};

static _Unwind_Reason_Code TraceFunction(_Unwind_Context* context, void* arg) {
  // The instruction pointer is pointing at the instruction after the return
  // call on all architectures.
  // Modify the pc to point at the real function.
  uintptr_t ip = _Unwind_GetIP(context);
  if (ip != 0) {
#if defined(__arm__)
    // If the ip is suspiciously low, do nothing to avoid a segfault trying
    // to access this memory.
    if (ip >= 4096) {
      // Check bits [15:11] of the first halfword assuming the instruction
      // is 32 bits long. If the bits are any of these values, then our
      // assumption was correct:
      //  b11101
      //  b11110
      //  b11111
      // Otherwise, this is a 16 bit instruction.
      uint16_t value = (*reinterpret_cast<uint16_t*>(ip - 2)) >> 11;
      if (value == 0x1f || value == 0x1e || value == 0x1d) {
        ip -= 4;
      } else {
        ip -= 2;
      }
    }
#elif defined(__aarch64__)
    // All instructions are 4 bytes long, skip back one instruction.
    ip -= 4;
#elif defined(__riscv)
    // C instructions are the shortest at 2 bytes long. (Unlike thumb, it's
    // non-trivial to recognize C instructions when going backwards in the
    // instruction stream.)
    ip -= 2;
#elif defined(__i386__) || defined(__x86_64__)
    // It's difficult to decode exactly where the previous instruction is,
    // so subtract 1 to estimate where the instruction lives.
    ip--;
#endif
  }

  StackState* state = static_cast<StackState*>(arg);
  state->frames[state->cur_frame++] = reinterpret_cast<void*>(ip);
  return (state->cur_frame >= state->frame_count) ? _URC_END_OF_STACK : _URC_NO_REASON;
}

int backtrace(void** buffer, int size) {
  if (size <= 0) {
    return 0;
  }

  StackState state(buffer, size);
  _Unwind_Backtrace(TraceFunction, &state);
  return state.cur_frame;
}

char** backtrace_symbols(void* const* buffer, int size) {
  if (size <= 0) {
    return nullptr;
  }
  // Do this calculation first in case the user passes in a bad value.
  size_t ptr_size;
  if (__builtin_mul_overflow(sizeof(char*), size, &ptr_size)) {
    return nullptr;
  }

  ScopedFd fd(memfd_create("backtrace_symbols_fd", MFD_CLOEXEC));
  if (fd.get() == -1) {
    return nullptr;
  }
  backtrace_symbols_fd(buffer, size, fd.get());

  // Get the size of the file.
  off_t file_size = lseek(fd.get(), 0, SEEK_END);
  if (file_size <= 0) {
    return nullptr;
  }

  // The interface for backtrace_symbols indicates that only the single
  // returned pointer must be freed by the caller. Therefore, allocate a
  // buffer that includes the memory for the strings and all of the pointers.
  // Add one byte at the end just in case the file didn't end with a '\n'.
  size_t symbol_data_size;
  if (__builtin_add_overflow(ptr_size, file_size, &symbol_data_size) ||
      __builtin_add_overflow(symbol_data_size, 1, &symbol_data_size)) {
    return nullptr;
  }

  uint8_t* symbol_data = reinterpret_cast<uint8_t*>(malloc(symbol_data_size));
  if (symbol_data == nullptr) {
    return nullptr;
  }

  // Copy the string data into the buffer.
  char* cur_string = reinterpret_cast<char*>(&symbol_data[ptr_size]);
  // If this fails, the read won't read back the correct number of bytes.
  lseek(fd.get(), 0, SEEK_SET);
  ssize_t num_read = read(fd.get(), cur_string, file_size);
  fd.reset(-1);
  if (num_read != file_size) {
    free(symbol_data);
    return nullptr;
  }

  // Make sure the last character in the file is '\n'.
  if (cur_string[file_size] != '\n') {
    cur_string[file_size++] = '\n';
  }

  for (int i = 0; i < size; i++) {
    (reinterpret_cast<char**>(symbol_data))[i] = cur_string;
    cur_string = strchr(cur_string, '\n');
    if (cur_string == nullptr) {
      free(symbol_data);
      return nullptr;
    }
    cur_string[0] = '\0';
    cur_string++;
  }
  return reinterpret_cast<char**>(symbol_data);
}

// This function should do no allocations if possible.
void backtrace_symbols_fd(void* const* buffer, int size, int fd) {
  if (size <= 0 || fd < 0) {
    return;
  }

  for (int frame_num = 0; frame_num < size; frame_num++) {
    void* address = buffer[frame_num];
    Dl_info info;
    if (dladdr(address, &info) != 0) {
      if (info.dli_fname != nullptr) {
        write(fd, info.dli_fname, strlen(info.dli_fname));
      }
      if (info.dli_sname != nullptr) {
        dprintf(fd, "(%s+0x%" PRIxPTR ") ", info.dli_sname,
                reinterpret_cast<uintptr_t>(address) - reinterpret_cast<uintptr_t>(info.dli_saddr));
      } else {
        dprintf(fd, "(+%p) ", info.dli_saddr);
      }
    }

    dprintf(fd, "[%p]\n", address);
  }
}
