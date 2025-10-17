/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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

#include <mutex>
#include <set>
#include <string>

#include <platform/bionic/macros.h>

class MapEntry {
 public:
  MapEntry() = default;
  MapEntry(uintptr_t start, uintptr_t end, uintptr_t offset, const char* name, size_t name_len,
           int flags)
      : start_(start), end_(end), offset_(offset), name_(name, name_len), flags_(flags) {}

  explicit MapEntry(uintptr_t pc) : start_(pc), end_(pc) {}

  void Init();

  uintptr_t GetLoadBias();

  void SetInvalid() {
    valid_ = false;
    init_ = true;
    load_bias_read_ = true;
  }

  bool valid() { return valid_; }
  uintptr_t start() const { return start_; }
  uintptr_t end() const { return end_; }
  uintptr_t offset() const { return offset_; }
  uintptr_t elf_start_offset() const { return elf_start_offset_; }
  void set_elf_start_offset(uintptr_t elf_start_offset) { elf_start_offset_ = elf_start_offset; }
  const std::string& name() const { return name_; }
  int flags() const { return flags_; }

 private:
  uintptr_t start_;
  uintptr_t end_;
  uintptr_t offset_;
  uintptr_t load_bias_ = 0;
  uintptr_t elf_start_offset_ = 0;
  std::string name_;
  int flags_;
  bool init_ = false;
  bool valid_ = false;
  bool load_bias_read_ = false;
};

// Ordering comparator that returns equivalence for overlapping entries
struct compare_entries {
  bool operator()(const MapEntry* a, const MapEntry* b) const { return a->end() <= b->start(); }
};

class MapData {
 public:
  MapData() = default;
  ~MapData();

  const MapEntry* find(uintptr_t pc, uintptr_t* rel_pc = nullptr);

  size_t NumMaps() { return entries_.size(); }

  void ReadMaps();

 private:
  std::mutex m_;
  std::set<MapEntry*, compare_entries> entries_;

  void ClearEntries();

  BIONIC_DISALLOW_COPY_AND_ASSIGN(MapData);
};
