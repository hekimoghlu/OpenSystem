/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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

#include <sys/cdefs.h>
#include <stddef.h>
#include <stdint.h>

const uint32_t kSmallObjectMaxSizeLog2 = 10;
const uint32_t kSmallObjectMinSizeLog2 = 4;
const uint32_t kSmallObjectAllocatorsCount = kSmallObjectMaxSizeLog2 - kSmallObjectMinSizeLog2 + 1;

class BionicSmallObjectAllocator;

// This structure is placed at the beginning of each addressable page
// and has all information we need to find the corresponding memory allocator.
struct page_info {
  char signature[4];
  uint32_t type;
  union {
    // we use allocated_size for large objects allocator
    size_t allocated_size;
    // and allocator_addr for small ones.
    BionicSmallObjectAllocator* allocator_addr;
  };
};

struct small_object_block_record {
  small_object_block_record* next;
  size_t free_blocks_cnt;
};

// This structure is placed at the beginning of each page managed by
// BionicSmallObjectAllocator.  Note that a page_info struct is expected at the
// beginning of each page as well, and therefore this structure contains a
// page_info as its *first* field.
struct small_object_page_info {
  page_info info;  // Must be the first field.

  // Doubly linked list for traversing all pages allocated by a
  // BionicSmallObjectAllocator.
  small_object_page_info* next_page;
  small_object_page_info* prev_page;

  // Linked list containing all free blocks in this page.
  small_object_block_record* free_block_list;

  // Free blocks counter.
  size_t free_blocks_cnt;
};

class BionicSmallObjectAllocator {
 public:
  BionicSmallObjectAllocator(uint32_t type, size_t block_size);
  void* alloc();
  void free(void* ptr);

  size_t get_block_size() const { return block_size_; }
 private:
  void alloc_page();
  void free_page(small_object_page_info* page);
  void add_to_page_list(small_object_page_info* page);
  void remove_from_page_list(small_object_page_info* page);

  const uint32_t type_;
  const size_t block_size_;
  const size_t blocks_per_page_;

  size_t free_pages_cnt_;

  small_object_page_info* page_list_;
};

class BionicAllocator {
 public:
  constexpr BionicAllocator() : allocators_(nullptr), allocators_buf_() {}
  void* alloc(size_t size);
  void* memalign(size_t align, size_t size);

  // Note that this implementation of realloc never shrinks allocation
  void* realloc(void* ptr, size_t size);
  void free(void* ptr);

  // Returns the size of the given allocated heap chunk, if it is valid.
  // Otherwise, this may return 0 or cause a segfault if the pointer is invalid.
  size_t get_chunk_size(void* ptr);

 private:
  void* alloc_mmap(size_t align, size_t size);
  inline void* alloc_impl(size_t align, size_t size);
  inline page_info* get_page_info_unchecked(void* ptr);
  inline page_info* get_page_info(void* ptr);
  BionicSmallObjectAllocator* get_small_object_allocator_unchecked(uint32_t type);
  BionicSmallObjectAllocator* get_small_object_allocator(page_info* pi, void* ptr);
  void initialize_allocators();

  BionicSmallObjectAllocator* allocators_;
  uint8_t allocators_buf_[sizeof(BionicSmallObjectAllocator)*kSmallObjectAllocatorsCount];
};
