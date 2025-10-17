/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 5, 2021.
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
#pragma once

#include <stdlib.h>
#include <limits.h>

#include <android-base/macros.h>

static constexpr size_t kBlockSizeAlign = sizeof(void*);
static constexpr size_t kBlockSizeMin = sizeof(void*) * 2;

struct LinkerBlockAllocatorPage;

/*
 * This class is a non-template version of the LinkerTypeAllocator
 * It keeps code inside .cpp file by keeping the interface
 * template-free.
 *
 * Please use LinkerTypeAllocator<type> where possible (everywhere).
 */
class LinkerBlockAllocator {
 public:
  explicit LinkerBlockAllocator(size_t block_size);

  void* alloc();
  void free(void* block);
  void protect_all(int prot);

  // Purge all pages if all previously allocated blocks have been freed.
  void purge();

 private:
  void create_new_page();
  LinkerBlockAllocatorPage* find_page(void* block);

  size_t block_size_;
  LinkerBlockAllocatorPage* page_list_;
  void* free_block_list_;
  size_t allocated_;

  DISALLOW_COPY_AND_ASSIGN(LinkerBlockAllocator);
};

/*
 * A simple allocator for the dynamic linker. An allocator allocates instances
 * of a single fixed-size type. Allocations are backed by page-sized private
 * anonymous mmaps.
 *
 * The differences between this allocator and BionicAllocator are:
 * 1. This allocator manages space more efficiently. BionicAllocator operates in
 *    power-of-two sized blocks up to 1k, when this implementation splits the
 *    page to aligned size of structure; For example for structures with size
 *    513 this allocator will use 516 (520 for lp64) bytes of data where
 *    generalized implementation is going to use 1024 sized blocks.
 *
 * 2. This allocator does not munmap allocated memory, where BionicAllocator does.
 *
 * 3. This allocator provides mprotect services to the user, where BionicAllocator
 *    always treats its memory as READ|WRITE.
 */
template<typename T>
class LinkerTypeAllocator {
 public:
  LinkerTypeAllocator() : block_allocator_(sizeof(T)) {}
  T* alloc() { return reinterpret_cast<T*>(block_allocator_.alloc()); }
  void free(T* t) { block_allocator_.free(t); }
  void protect_all(int prot) { block_allocator_.protect_all(prot); }
 private:
  LinkerBlockAllocator block_allocator_;
  DISALLOW_COPY_AND_ASSIGN(LinkerTypeAllocator);
};
