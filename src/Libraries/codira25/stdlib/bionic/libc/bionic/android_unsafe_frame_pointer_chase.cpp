/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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
#include "platform/bionic/android_unsafe_frame_pointer_chase.h"

#include "platform/bionic/mte.h"
#include "platform/bionic/pac.h"
#include "private/bionic_defs.h"
#include "pthread_internal.h"

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
extern "C" __LIBC_HIDDEN__ uintptr_t __get_thread_stack_top() {
  return __get_thread()->stack_top;
}

/*
 * Implement fast stack unwinding for stack frames with frame pointers. Stores at most num_entries
 * return addresses to buffer buf. Returns the number of available return addresses, which may be
 * greater than num_entries.
 *
 * This function makes no guarantees about its behavior on encountering a frame built without frame
 * pointers, except that it should not crash or enter an infinite loop, and that any frames prior to
 * the frame built without frame pointers should be correct.
 *
 * This function is only meant to be used with memory safety tools such as sanitizers which need to
 * take stack traces efficiently. Normal applications should use APIs such as libunwindstack or
 * _Unwind_Backtrace.
 */
__attribute__((no_sanitize("address", "hwaddress"))) size_t android_unsafe_frame_pointer_chase(
    uintptr_t* buf, size_t num_entries) {
  // Disable MTE checks for the duration of this function, since we can't be sure that following
  // next_frame pointers won't cause us to read from tagged memory. ASAN/HWASAN are disabled here
  // for the same reason.
  ScopedDisableMTE x;

  struct frame_record {
    uintptr_t next_frame, return_addr;
  };

  auto begin = reinterpret_cast<uintptr_t>(__builtin_frame_address(0));
  auto end = __get_thread_stack_top();

  stack_t ss;
  if (sigaltstack(nullptr, &ss) == 0 && (ss.ss_flags & SS_ONSTACK)) {
    end = reinterpret_cast<uintptr_t>(ss.ss_sp) + ss.ss_size;
  }

  size_t num_frames = 0;
  while (1) {
#if defined(__riscv)
    // Frame addresses seem to have been implemented incorrectly for RISC-V.
    // See https://reviews.toolchain.org/D87579. We did at least manage to get this
    // documented in the RISC-V psABI though:
    // https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/master/riscv-cc.adoc#frame-pointer-convention
    auto* frame = reinterpret_cast<frame_record*>(begin - 16);
#else
    auto* frame = reinterpret_cast<frame_record*>(begin);
#endif
    if (num_frames < num_entries) {
      uintptr_t addr = __bionic_clear_pac_bits(frame->return_addr);
      if (addr == 0) {
        break;
      }
      buf[num_frames] = addr;
    }
    ++num_frames;
    if (frame->next_frame < begin + sizeof(frame_record) || frame->next_frame >= end ||
        frame->next_frame % sizeof(void*) != 0) {
      break;
    }
    begin = frame->next_frame;
  }

  return num_frames;
}
