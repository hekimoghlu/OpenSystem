/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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
#include "linker_block_allocator.h"

#include <inttypes.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/prctl.h>
#include <unistd.h>

#include "linker_debug.h"

static constexpr size_t kMaxPageSize = 65536;
static constexpr size_t kAllocateSize = kMaxPageSize * 6;
static_assert(kAllocateSize % kMaxPageSize == 0, "Invalid kAllocateSize.");

struct LinkerBlockAllocatorPage {
  LinkerBlockAllocatorPage* next;
  uint8_t bytes[kAllocateSize - 16] __attribute__((aligned(16)));
};

struct FreeBlockInfo {
  void* next_block;
  size_t num_free_blocks;
};

static_assert(kBlockSizeAlign >= alignof(FreeBlockInfo));
static_assert(kBlockSizeMin == sizeof(FreeBlockInfo));

LinkerBlockAllocator::LinkerBlockAllocator(size_t block_size)
    : block_size_(__BIONIC_ALIGN(MAX(block_size, kBlockSizeMin), kBlockSizeAlign)),
      page_list_(nullptr),
      free_block_list_(nullptr),
      allocated_(0) {}

void* LinkerBlockAllocator::alloc() {
  if (free_block_list_ == nullptr) {
    create_new_page();
  }

  FreeBlockInfo* block_info = reinterpret_cast<FreeBlockInfo*>(free_block_list_);
  if (block_info->num_free_blocks > 1) {
    FreeBlockInfo* next_block_info = reinterpret_cast<FreeBlockInfo*>(
      reinterpret_cast<char*>(free_block_list_) + block_size_);
    next_block_info->next_block = block_info->next_block;
    next_block_info->num_free_blocks = block_info->num_free_blocks - 1;
    free_block_list_ = next_block_info;
  } else {
    free_block_list_ = block_info->next_block;
  }

  memset(block_info, 0, block_size_);

  ++allocated_;

  return block_info;
}

void LinkerBlockAllocator::free(void* block) {
  if (block == nullptr) {
    return;
  }

  LinkerBlockAllocatorPage* page = find_page(block);
  CHECK(page != nullptr);

  ssize_t offset = reinterpret_cast<uint8_t*>(block) - page->bytes;
  CHECK((offset % block_size_) == 0);

  memset(block, 0, block_size_);

  FreeBlockInfo* block_info = reinterpret_cast<FreeBlockInfo*>(block);

  block_info->next_block = free_block_list_;
  block_info->num_free_blocks = 1;

  free_block_list_ = block_info;

  --allocated_;
}

void LinkerBlockAllocator::protect_all(int prot) {
  for (LinkerBlockAllocatorPage* page = page_list_; page != nullptr; page = page->next) {
    if (mprotect(page, kAllocateSize, prot) == -1) {
      async_safe_fatal("mprotect(%p, %zu, %d) failed: %m", page, kAllocateSize, prot);
    }
  }
}

void LinkerBlockAllocator::create_new_page() {
  static_assert(sizeof(LinkerBlockAllocatorPage) == kAllocateSize,
                "Invalid sizeof(LinkerBlockAllocatorPage)");

  LinkerBlockAllocatorPage* page = reinterpret_cast<LinkerBlockAllocatorPage*>(
      mmap(nullptr, kAllocateSize, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0));
  CHECK(page != MAP_FAILED);

  prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, page, kAllocateSize, "linker_alloc");

  FreeBlockInfo* first_block = reinterpret_cast<FreeBlockInfo*>(page->bytes);
  first_block->next_block = free_block_list_;
  first_block->num_free_blocks = sizeof(page->bytes) / block_size_;

  free_block_list_ = first_block;

  page->next = page_list_;
  page_list_ = page;
}

LinkerBlockAllocatorPage* LinkerBlockAllocator::find_page(void* block) {
  CHECK(block != nullptr);

  LinkerBlockAllocatorPage* page = page_list_;
  while (page != nullptr) {
    const uint8_t* page_ptr = reinterpret_cast<const uint8_t*>(page);
    if (block >= (page_ptr + sizeof(page->next)) && block < (page_ptr + kAllocateSize)) {
      return page;
    }

    page = page->next;
  }

  async_safe_fatal("couldn't find page for %p", block);
}

void LinkerBlockAllocator::purge() {
  if (allocated_) {
    return;
  }

  LinkerBlockAllocatorPage* page = page_list_;
  while (page) {
    LinkerBlockAllocatorPage* next = page->next;
    munmap(page, kAllocateSize);
    page = next;
  }
  page_list_ = nullptr;
  free_block_list_ = nullptr;
}
