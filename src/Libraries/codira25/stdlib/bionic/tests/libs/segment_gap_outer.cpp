/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 20, 2025.
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

#include <android/dlext.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <unistd.h>

extern "C" void __attribute__((section(".custom_text"))) text_before_start_of_gap() {}
char __attribute__((section(".custom_bss"))) end_of_gap[0x1000];

extern "C" void* get_inner() {
  android_dlextinfo info = {};
  info.flags = ANDROID_DLEXT_RESERVED_ADDRESS;

  char* start_of_gap =
      reinterpret_cast<char*>(
          (reinterpret_cast<uintptr_t>(text_before_start_of_gap) &
           ~(sysconf(_SC_PAGESIZE) - 1)) + sysconf(_SC_PAGESIZE));
  info.reserved_addr = start_of_gap;
  info.reserved_size = end_of_gap - start_of_gap;

  void *handle = android_dlopen_ext("libsegment_gap_inner.so", RTLD_NOW, &info);
  if (!handle) {
    __builtin_trap();
  }

  return dlsym(handle, "inner");
}
