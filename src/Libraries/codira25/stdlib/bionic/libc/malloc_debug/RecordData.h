/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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

#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <unistd.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <memory_trace/MemoryTrace.h>
#include <platform/bionic/macros.h>

class Config;

class RecordData {
 public:
  RecordData();
  virtual ~RecordData();

  bool Initialize(const Config& config);

  memory_trace::Entry* ReserveEntry();

  const std::string& file() { return file_; }
  pthread_key_t key() { return key_; }

  int64_t GetPresentBytes(void* pointer, size_t size);

  static void WriteEntriesOnExit();

 private:
  static void WriteData(int, siginfo_t*, void*);
  static RecordData* record_obj_;

  static void ThreadKeyDelete(void* data);

  void WriteEntries();
  void WriteEntries(const std::string& file);

  memory_trace::Entry* InternalReserveEntry();

  std::mutex entries_lock_;
  pthread_key_t key_;
  std::vector<memory_trace::Entry> entries_;
  size_t cur_index_;
  std::string file_;
  int pagemap_fd_ = -1;

  BIONIC_DISALLOW_COPY_AND_ASSIGN(RecordData);
};
