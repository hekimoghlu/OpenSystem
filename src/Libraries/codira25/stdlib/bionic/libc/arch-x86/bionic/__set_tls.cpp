/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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
#include <limits.h>
#include <pthread.h>

#include <asm/ldt.h>

extern "C" int __set_thread_area(user_desc*);

__LIBC_HIDDEN__ void __init_user_desc(user_desc* result, bool allocate, void* base_addr) {
  if (allocate) {
    // Let the kernel choose.
    result->entry_number = -1;
  } else {
    // Get the existing entry number from %gs.
    uint32_t gs;
    __asm__ __volatile__("movw %%gs, %w0" : "=q"(gs) /*output*/);
    result->entry_number = (gs & 0xffff) >> 3;
  }

  result->base_addr = reinterpret_cast<uintptr_t>(base_addr);

  result->limit = 0xfffff;

  result->seg_32bit = 1;
  result->contents = MODIFY_LDT_CONTENTS_DATA;
  result->read_exec_only = 0;
  result->limit_in_pages = 1;
  result->seg_not_present = 0;
  result->useable = 1;
}

extern "C" __LIBC_HIDDEN__ int __set_tls(void* ptr) {
  user_desc tls_descriptor = {};
  __init_user_desc(&tls_descriptor, true, ptr);

  int rc = __set_thread_area(&tls_descriptor);
  if (rc != -1) {
    // Change %gs to be new GDT entry.
    uint16_t table_indicator = 0;  // GDT
    uint16_t rpl = 3;  // Requested privilege level
    uint16_t selector = (tls_descriptor.entry_number << 3) | table_indicator | rpl;
    __asm__ __volatile__("movw %w0, %%gs" : /*output*/ : "q"(selector) /*input*/ : /*clobber*/);
  }

  return rc;
}
