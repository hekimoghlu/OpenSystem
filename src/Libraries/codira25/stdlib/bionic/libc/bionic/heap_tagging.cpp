/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 26, 2024.
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
#include "heap_tagging.h"
#include "malloc_common.h"
#include "malloc_tagged_pointers.h"

#include <bionic/pthread_internal.h>
#include <platform/bionic/malloc.h>
#include <sanitizer/hwasan_interface.h>
#include <sys/auxv.h>
#include <sys/prctl.h>

extern "C" void h_malloc_disable_memory_tagging();

extern "C" void scudo_malloc_disable_memory_tagging();
extern "C" void scudo_malloc_set_track_allocation_stacks(int);

extern "C" const char* __scudo_get_stack_depot_addr();
extern "C" const char* __scudo_get_ring_buffer_addr();
extern "C" size_t __scudo_get_ring_buffer_size();
extern "C" size_t __scudo_get_stack_depot_size();

// Protected by `g_heap_tagging_lock`.
static HeapTaggingLevel heap_tagging_level = M_HEAP_TAGGING_LEVEL_NONE;

void SetDefaultHeapTaggingLevel() {
#if defined(__aarch64__)
#if !__has_feature(hwaddress_sanitizer)
  heap_tagging_level = __libc_shared_globals()->initial_heap_tagging_level;
#endif

  __libc_memtag_stack_abi = __libc_shared_globals()->initial_memtag_stack_abi;

  __libc_globals.mutate([](libc_globals* globals) {
    switch (heap_tagging_level) {
      case M_HEAP_TAGGING_LEVEL_TBI:
        // Arrange for us to set pointer tags to POINTER_TAG, check tags on
        // deallocation and untag when passing pointers to the allocator.
        globals->heap_pointer_tag = (reinterpret_cast<uintptr_t>(POINTER_TAG) << TAG_SHIFT) |
                                    (0xffull << CHECK_SHIFT) | (0xffull << UNTAG_SHIFT);
        break;
      case M_HEAP_TAGGING_LEVEL_SYNC:
      case M_HEAP_TAGGING_LEVEL_ASYNC:
        atomic_store(&globals->memtag, true);
        atomic_store(&__libc_memtag_stack, __libc_shared_globals()->initial_memtag_stack);
        break;
      default:
        break;
    };
  });


  switch (heap_tagging_level) {
    case M_HEAP_TAGGING_LEVEL_TBI:
    case M_HEAP_TAGGING_LEVEL_NONE:
#if defined(USE_SCUDO)
      scudo_malloc_disable_memory_tagging();
#endif
#if defined(USE_H_MALLOC)
      h_malloc_disable_memory_tagging();
#endif
      break;
    case M_HEAP_TAGGING_LEVEL_SYNC:
#if defined(USE_SCUDO)
      scudo_malloc_set_track_allocation_stacks(1);
#endif
      break;
    default:
      break;
  }

#endif  // aarch64
}

static bool set_tcf_on_all_threads(int tcf) {
  return android_run_on_all_threads(
      [](void* arg) {
        int tcf = *reinterpret_cast<int*>(arg);
        int tagged_addr_ctrl = prctl(PR_GET_TAGGED_ADDR_CTRL, 0, 0, 0, 0);
        if (tagged_addr_ctrl < 0) {
          return false;
        }

        tagged_addr_ctrl = (tagged_addr_ctrl & ~PR_MTE_TCF_MASK) | tcf;
        return prctl(PR_SET_TAGGED_ADDR_CTRL, tagged_addr_ctrl, 0, 0, 0) >= 0;
      },
      &tcf);
}

pthread_mutex_t g_heap_tagging_lock = PTHREAD_MUTEX_INITIALIZER;

static bool block_heap_tagging_level_downgrade;

// Requires `g_heap_tagging_lock` to be held.
bool BlockHeapTaggingLevelDowngrade() {
  if (block_heap_tagging_level_downgrade) {
    return false;
  }
  block_heap_tagging_level_downgrade = true;
  return true;
}

// Requires `g_heap_tagging_lock` to be held.
bool SetHeapTaggingLevel(HeapTaggingLevel tag_level) {
  if (tag_level == heap_tagging_level) {
    return true;
  }

  if (block_heap_tagging_level_downgrade) {
    // allow switching between SYNC and ASYNC, but don't allow disabling memory tagging
    if (tag_level < heap_tagging_level && tag_level != M_HEAP_TAGGING_LEVEL_ASYNC) {
      error_log("SetHeapTaggingLevel: blocked downgrade of tag level from %i to %i", heap_tagging_level, tag_level);
      return false;
    }
  }

  switch (tag_level) {
    case M_HEAP_TAGGING_LEVEL_NONE:
      __libc_globals.mutate([](libc_globals* globals) {
        if (heap_tagging_level == M_HEAP_TAGGING_LEVEL_TBI) {
          // Preserve the untag mask (we still want to untag pointers when passing them to the
          // allocator), but clear the fixed tag and the check mask, so that pointers are no longer
          // tagged and checks no longer happen.
          globals->heap_pointer_tag = static_cast<uintptr_t>(0xffull << UNTAG_SHIFT);
        }
        atomic_store(&__libc_memtag_stack, false);
        atomic_store(&globals->memtag, false);
        atomic_store(&__libc_shared_globals()->memtag_currently_on, false);
      });

      if (heap_tagging_level != M_HEAP_TAGGING_LEVEL_TBI) {
        if (!set_tcf_on_all_threads(PR_MTE_TCF_NONE)) {
          error_log("SetHeapTaggingLevel: set_tcf_on_all_threads failed");
          return false;
        }
      }
#if defined(USE_SCUDO) && !__has_feature(hwaddress_sanitizer)
      scudo_malloc_disable_memory_tagging();
#endif
#if defined(USE_H_MALLOC)
      h_malloc_disable_memory_tagging();
#endif
      break;
    case M_HEAP_TAGGING_LEVEL_TBI:
    case M_HEAP_TAGGING_LEVEL_ASYNC:
    case M_HEAP_TAGGING_LEVEL_SYNC:
      if (heap_tagging_level == M_HEAP_TAGGING_LEVEL_NONE) {
#if !__has_feature(hwaddress_sanitizer)
        // Suppress the error message in HWASan builds. Apps can try to enable TBI (or even MTE
        // modes) being unaware of HWASan, fail them silently.
        error_log(
            "SetHeapTaggingLevel: re-enabling tagging after it was disabled is not supported");
#endif
        return false;
      } else if (tag_level == M_HEAP_TAGGING_LEVEL_TBI ||
                 heap_tagging_level == M_HEAP_TAGGING_LEVEL_TBI) {
        error_log("SetHeapTaggingLevel: switching between TBI and ASYNC/SYNC is not supported");
        return false;
      }

      if (tag_level == M_HEAP_TAGGING_LEVEL_ASYNC) {
        // When entering ASYNC mode, specify that we want to allow upgrading to SYNC by OR'ing in
        // the SYNC flag. But if the kernel doesn't support specifying multiple TCF modes, fall back
        // to specifying a single mode.
        if (!set_tcf_on_all_threads(PR_MTE_TCF_ASYNC | PR_MTE_TCF_SYNC)) {
          set_tcf_on_all_threads(PR_MTE_TCF_ASYNC);
        }
#if defined(USE_SCUDO) && !__has_feature(hwaddress_sanitizer)
        scudo_malloc_set_track_allocation_stacks(0);
#endif
      } else if (tag_level == M_HEAP_TAGGING_LEVEL_SYNC) {
        set_tcf_on_all_threads(PR_MTE_TCF_SYNC);
#if defined(USE_SCUDO) && !__has_feature(hwaddress_sanitizer)
        scudo_malloc_set_track_allocation_stacks(1);
        __libc_shared_globals()->scudo_ring_buffer = __scudo_get_ring_buffer_addr();
        __libc_shared_globals()->scudo_ring_buffer_size = __scudo_get_ring_buffer_size();
        __libc_shared_globals()->scudo_stack_depot = __scudo_get_stack_depot_addr();
        __libc_shared_globals()->scudo_stack_depot_size = __scudo_get_stack_depot_size();
#endif
      }
      break;
    default:
      error_log("SetHeapTaggingLevel: unknown tagging level");
      return false;
  }

  heap_tagging_level = tag_level;
  info_log("SetHeapTaggingLevel: tag level set to %d", tag_level);

  return true;
}

#ifdef __aarch64__
static inline __attribute__((no_sanitize("memtag"))) void untag_memory(void* from, void* to) {
  if (from == to) {
    return;
  }
  __asm__ __volatile__(
      ".arch_extension mte\n"
      "1:\n"
      "stg %[Ptr], [%[Ptr]], #16\n"
      "cmp %[Ptr], %[End]\n"
      "b.lt 1b\n"
      : [Ptr] "+&r"(from)
      : [End] "r"(to)
      : "memory");
}
#endif

#ifdef __aarch64__
// 128Mb of stack should be enough for anybody.
static constexpr size_t kUntagLimit = 128 * 1024 * 1024;
#endif  // __aarch64__

extern "C" __LIBC_HIDDEN__ __attribute__((no_sanitize("memtag"))) void memtag_handle_longjmp(
    void* sp_dst __unused, void* sp_src __unused) {
  // A usual longjmp looks like this, where sp_dst was the LR in the call to setlongjmp (i.e.
  // the SP of the frame calling setlongjmp).
  // â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                  â”‚
  // â”‚                     â”‚                  â”‚
  // â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤â—„â”€â”€â”€â”€â”€â”€â”€â”€ sp_dst  â”‚ stack
  // â”‚         ...         â”‚                  â”‚ grows
  // â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤                  â”‚ to lower
  // â”‚         ...         â”‚                  â”‚ addresses
  // â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤â—„â”€â”€â”€â”€â”€â”€â”€â”€ sp_src  â”‚
  // â”‚siglongjmp           â”‚                  â”‚
  // â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤                  â”‚
  // â”‚memtag_handle_longjmpâ”‚                  â”‚
  // â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                  â–¼
#ifdef __aarch64__
  if (atomic_load(&__libc_memtag_stack)) {
    size_t distance = reinterpret_cast<uintptr_t>(sp_dst) - reinterpret_cast<uintptr_t>(sp_src);
    if (distance > kUntagLimit) {
      async_safe_fatal(
          "memtag_handle_longjmp: stack adjustment too large! %p -> %p, distance %zx > %zx\n",
          sp_src, sp_dst, distance, kUntagLimit);
    } else {
      untag_memory(sp_src, sp_dst);
    }
  }
#endif  // __aarch64__

  // We can use __has_feature here rather than __hwasan_handle_longjmp as a
  // weak symbol because this is part of libc which is always sanitized for a
  // hwasan enabled process.
#if __has_feature(hwaddress_sanitizer)
  __hwasan_handle_longjmp(sp_dst);
#endif  // __has_feature(hwaddress_sanitizer)
}
