/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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

#include <stdint.h>
#include <stdio.h>

#include <atomic>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <platform/bionic/macros.h>
#include <unwindstack/Unwinder.h>

#include "OptionData.h"
#include "UnwindBacktrace.h"

extern bool* g_zygote_child;

// Forward declarations.
class Config;

struct FrameKeyType {
  size_t num_frames;
  uintptr_t* frames;

  bool operator==(const FrameKeyType& comp) const {
    if (num_frames != comp.num_frames) return false;
    for (size_t i = 0; i < num_frames; i++) {
      if (frames[i] != comp.frames[i]) {
        return false;
      }
    }
    return true;
  }
};

namespace std {
template <>
struct hash<FrameKeyType> {
  std::size_t operator()(const FrameKeyType& key) const {
    std::size_t cur_hash = key.frames[0];
    // Limit the number of frames to speed up hashing.
    size_t max_frames = (key.num_frames > 5) ? 5 : key.num_frames;
    for (size_t i = 1; i < max_frames; i++) {
      cur_hash ^= key.frames[i];
    }
    return cur_hash;
  }
};
};  // namespace std

struct FrameInfoType {
  size_t references = 0;
  std::vector<uintptr_t> frames;
};

struct PointerInfoType {
  size_t size;
  size_t hash_index;
  size_t RealSize() const { return size & ~(1U << 31); }
  bool ZygoteChildAlloc() const { return size & (1U << 31); }
  static size_t GetEncodedSize(size_t size) {
    return GetEncodedSize(*g_zygote_child, size);
  }
  static size_t GetEncodedSize(bool child_alloc, size_t size) {
    return size | ((child_alloc) ? (1U << 31) : 0);
  }
  static size_t MaxSize() { return (1U << 31) - 1; }
};

struct FreePointerInfoType {
  uintptr_t mangled_ptr;
  size_t hash_index;
};

struct ListInfoType {
  uintptr_t pointer;
  size_t num_allocations;
  size_t size;
  bool zygote_child_alloc;
  FrameInfoType* frame_info;
  std::vector<unwindstack::FrameData>* backtrace_info;
};

class PointerData : public OptionData {
 public:
  explicit PointerData(DebugData* debug_data);
  virtual ~PointerData() = default;

  bool Initialize(const Config& config);

  inline size_t alloc_offset() { return alloc_offset_; }

  bool ShouldBacktrace() { return backtrace_enabled_ == 1; }
  void ToggleBacktraceEnabled() { backtrace_enabled_.fetch_xor(1); }

  void EnableDumping() { backtrace_dump_ = true; }
  bool ShouldDumpAndReset() {
    bool expected = true;
    return backtrace_dump_.compare_exchange_strong(expected, false);
  }

  void PrepareFork();
  void PostForkParent();
  void PostForkChild();

  static void IteratePointers(std::function<void(uintptr_t pointer)> fn);

  static size_t AddBacktrace(size_t num_frames, size_t size_bytes);
  static void RemoveBacktrace(size_t hash_index);

  static void Add(const void* pointer, size_t size);
  static void Remove(const void* pointer);

  static void* AddFreed(const void* pointer, size_t size_bytes);
  static void LogFreeError(const FreePointerInfoType& info, size_t usable_size);
  static void LogFreeBacktrace(const void* ptr);
  static void VerifyFreedPointer(const FreePointerInfoType& info);
  static void VerifyAllFreed();

  static void GetAllocList(std::vector<ListInfoType>* list);
  static void LogLeaks();
  static void DumpLiveToFile(int fd);

  static void GetInfo(uint8_t** info, size_t* overall_size, size_t* info_size, size_t* total_memory,
                      size_t* backtrace_size);

  static size_t GetFrames(const void* pointer, uintptr_t* frames, size_t max_frames);

  static bool Exists(const void* pointer);

 private:
  // Only keep mangled pointers in internal data structures. This avoids
  // problems where libmemunreachable finds these pointers and thinks they
  // are not unreachable.
  static inline uintptr_t ManglePointer(uintptr_t pointer) { return pointer ^ UINTPTR_MAX; }
  static inline uintptr_t DemanglePointer(uintptr_t pointer) { return pointer ^ UINTPTR_MAX; }

  static std::string GetHashString(uintptr_t* frames, size_t num_frames);
  static void LogBacktrace(size_t hash_index);

  static void GetList(std::vector<ListInfoType>* list, bool only_with_backtrace);
  static void GetUniqueList(std::vector<ListInfoType>* list, bool only_with_backtrace);

  size_t alloc_offset_ = 0;
  std::vector<uint8_t> cmp_mem_;

  static std::atomic_uint8_t backtrace_enabled_;

  static std::atomic_bool backtrace_dump_;

  static std::mutex pointer_mutex_;
  static std::unordered_map<uintptr_t, PointerInfoType> pointers_;

  static std::mutex frame_mutex_;
  static std::unordered_map<FrameKeyType, size_t> key_to_index_;
  static std::unordered_map<size_t, FrameInfoType> frames_;
  static std::unordered_map<size_t, std::vector<unwindstack::FrameData>> backtraces_info_;
  static size_t cur_hash_index_;

  static std::mutex free_pointer_mutex_;
  static std::deque<FreePointerInfoType> free_pointers_;

  BIONIC_DISALLOW_COPY_AND_ASSIGN(PointerData);
};
