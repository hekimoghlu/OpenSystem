/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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
#include "rtc_base/memory/aligned_malloc.h"

#include <stdlib.h>  // for free, malloc
#include <string.h>  // for memcpy

#include "rtc_base/checks.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <stdint.h>
#endif

// Reference on memory alignment:
// http://stackoverflow.com/questions/227897/solve-the-memory-alignment-in-c-interview-question-that-stumped-me
namespace webrtc {

uintptr_t GetRightAlign(uintptr_t start_pos, size_t alignment) {
  // The pointer should be aligned with `alignment` bytes. The - 1 guarantees
  // that it is aligned towards the closest higher (right) address.
  return (start_pos + alignment - 1) & ~(alignment - 1);
}

// Alignment must be an integer power of two.
bool ValidAlignment(size_t alignment) {
  if (!alignment) {
    return false;
  }
  return (alignment & (alignment - 1)) == 0;
}

void* GetRightAlign(const void* pointer, size_t alignment) {
  if (!pointer) {
    return NULL;
  }
  if (!ValidAlignment(alignment)) {
    return NULL;
  }
  uintptr_t start_pos = reinterpret_cast<uintptr_t>(pointer);
  return reinterpret_cast<void*>(GetRightAlign(start_pos, alignment));
}

void* AlignedMalloc(size_t size, size_t alignment) {
  if (size == 0) {
    return NULL;
  }
  if (!ValidAlignment(alignment)) {
    return NULL;
  }

  // The memory is aligned towards the lowest address that so only
  // alignment - 1 bytes needs to be allocated.
  // A pointer to the start of the memory must be stored so that it can be
  // retreived for deletion, ergo the sizeof(uintptr_t).
  void* memory_pointer = malloc(size + sizeof(uintptr_t) + alignment - 1);
  RTC_CHECK(memory_pointer) << "Couldn't allocate memory in AlignedMalloc";

  // Aligning after the sizeof(uintptr_t) bytes will leave room for the header
  // in the same memory block.
  uintptr_t align_start_pos = reinterpret_cast<uintptr_t>(memory_pointer);
  align_start_pos += sizeof(uintptr_t);
  uintptr_t aligned_pos = GetRightAlign(align_start_pos, alignment);
  void* aligned_pointer = reinterpret_cast<void*>(aligned_pos);

  // Store the address to the beginning of the memory just before the aligned
  // memory.
  uintptr_t header_pos = aligned_pos - sizeof(uintptr_t);
  void* header_pointer = reinterpret_cast<void*>(header_pos);
  uintptr_t memory_start = reinterpret_cast<uintptr_t>(memory_pointer);
  memcpy(header_pointer, &memory_start, sizeof(uintptr_t));

  return aligned_pointer;
}

void AlignedFree(void* mem_block) {
  if (mem_block == NULL) {
    return;
  }
  uintptr_t aligned_pos = reinterpret_cast<uintptr_t>(mem_block);
  uintptr_t header_pos = aligned_pos - sizeof(uintptr_t);

  // Read out the address of the AlignedMemory struct from the header.
  uintptr_t memory_start_pos = *reinterpret_cast<uintptr_t*>(header_pos);
  void* memory_start = reinterpret_cast<void*>(memory_start_pos);
  free(memory_start);
}

}  // namespace webrtc
