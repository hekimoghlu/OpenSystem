/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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

#include <string>
#include <unordered_map>

constexpr uint64_t FRONT_GUARD = 0x1;
constexpr uint64_t REAR_GUARD = 0x2;
constexpr uint64_t BACKTRACE = 0x4;
constexpr uint64_t FILL_ON_ALLOC = 0x8;
constexpr uint64_t FILL_ON_FREE = 0x10;
constexpr uint64_t EXPAND_ALLOC = 0x20;
constexpr uint64_t FREE_TRACK = 0x40;
constexpr uint64_t TRACK_ALLOCS = 0x80;
constexpr uint64_t LEAK_TRACK = 0x100;
constexpr uint64_t RECORD_ALLOCS = 0x200;
constexpr uint64_t BACKTRACE_FULL = 0x400;
constexpr uint64_t ABORT_ON_ERROR = 0x800;
constexpr uint64_t VERBOSE = 0x1000;
constexpr uint64_t CHECK_UNREACHABLE_ON_SIGNAL = 0x2000;
constexpr uint64_t BACKTRACE_SPECIFIC_SIZES = 0x4000;
constexpr uint64_t LOG_ALLOCATOR_STATS_ON_SIGNAL = 0x8000;
constexpr uint64_t LOG_ALLOCATOR_STATS_ON_EXIT = 0x10000;

// In order to guarantee posix compliance, set the minimum alignment
// to 8 bytes for 32 bit systems and 16 bytes for 64 bit systems.
#if defined(__LP64__)
constexpr size_t MINIMUM_ALIGNMENT_BYTES = 16;
#else
constexpr size_t MINIMUM_ALIGNMENT_BYTES = 8;
#endif

// If one or more of these options is set, then a special header is needed.
constexpr uint64_t HEADER_OPTIONS = FRONT_GUARD | REAR_GUARD;

class Config {
 public:
  bool Init(const char* options_str);

  void LogUsage() const;

  uint64_t options() const { return options_; }

  int backtrace_signal() const { return backtrace_signal_; }
  int backtrace_dump_signal() const { return backtrace_dump_signal_; }
  size_t backtrace_frames() const { return backtrace_frames_; }
  size_t backtrace_enabled() const { return backtrace_enabled_; }
  size_t backtrace_enable_on_signal() const { return backtrace_enable_on_signal_; }
  bool backtrace_dump_on_exit() const { return backtrace_dump_on_exit_; }
  const std::string& backtrace_dump_prefix() const { return backtrace_dump_prefix_; }

  size_t front_guard_bytes() const { return front_guard_bytes_; }
  size_t rear_guard_bytes() const { return rear_guard_bytes_; }
  uint8_t front_guard_value() const { return front_guard_value_; }
  uint8_t rear_guard_value() const { return rear_guard_value_; }

  size_t expand_alloc_bytes() const { return expand_alloc_bytes_; }

  size_t free_track_allocations() const { return free_track_allocations_; }
  size_t free_track_backtrace_num_frames() const { return free_track_backtrace_num_frames_; }

  size_t fill_on_alloc_bytes() const { return fill_on_alloc_bytes_; }
  size_t fill_on_free_bytes() const { return fill_on_free_bytes_; }
  uint8_t fill_alloc_value() const { return fill_alloc_value_; }
  uint8_t fill_free_value() const { return fill_free_value_; }

  size_t backtrace_min_size_bytes() const { return backtrace_min_size_bytes_; }
  size_t backtrace_max_size_bytes() const { return backtrace_max_size_bytes_; }

  int record_allocs_signal() const { return record_allocs_signal_; }
  size_t record_allocs_num_entries() const { return record_allocs_num_entries_; }
  const std::string& record_allocs_file() const { return record_allocs_file_; }
  bool record_allocs_on_exit() const { return record_allocs_on_exit_; }

  int check_unreachable_signal() const { return check_unreachable_signal_; }

  int log_allocator_stats_signal() const { return log_allocator_stats_signal_; }

 private:
  struct OptionInfo {
    uint64_t option;
    bool (Config::*process_func)(const std::string&, const std::string&);
  };

  bool ParseValue(const std::string& option, const std::string& value, size_t default_value,
                  size_t min_value, size_t max_value, size_t* new_value) const;

  bool ParseValue(const std::string& option, const std::string& value, size_t min_value,
                  size_t max_value, size_t* parsed_value) const;

  bool SetGuard(const std::string& option, const std::string& value);
  bool SetFrontGuard(const std::string& option, const std::string& value);
  bool SetRearGuard(const std::string& option, const std::string& value);

  bool SetFill(const std::string& option, const std::string& value);
  bool SetFillOnAlloc(const std::string& option, const std::string& value);
  bool SetFillOnFree(const std::string& option, const std::string& value);

  bool SetBacktrace(const std::string& option, const std::string& value);
  bool SetBacktraceEnableOnSignal(const std::string& option, const std::string& value);
  bool SetBacktraceDumpOnExit(const std::string& option, const std::string& value);
  bool SetBacktraceDumpPrefix(const std::string& option, const std::string& value);

  bool SetBacktraceSize(const std::string& option, const std::string& value);
  bool SetBacktraceMinSize(const std::string& option, const std::string& value);
  bool SetBacktraceMaxSize(const std::string& option, const std::string& value);

  bool SetExpandAlloc(const std::string& option, const std::string& value);

  bool SetFreeTrack(const std::string& option, const std::string& value);
  bool SetFreeTrackBacktraceNumFrames(const std::string& option, const std::string& value);

  bool SetRecordAllocs(const std::string& option, const std::string& value);
  bool SetRecordAllocsFile(const std::string& option, const std::string& value);
  bool SetRecordAllocsOnExit(const std::string& option, const std::string& value);

  bool VerifyValueEmpty(const std::string& option, const std::string& value);

  static bool GetOption(const char** option_str, std::string* option, std::string* value);

  const static std::unordered_map<std::string, OptionInfo> kOptions;

  size_t front_guard_bytes_ = 0;
  size_t rear_guard_bytes_ = 0;

  bool backtrace_enable_on_signal_ = false;
  int backtrace_signal_ = 0;
  int backtrace_dump_signal_ = 0;
  bool backtrace_enabled_ = false;
  size_t backtrace_frames_ = 0;
  bool backtrace_dump_on_exit_ = false;
  std::string backtrace_dump_prefix_;
  size_t backtrace_min_size_bytes_ = 0;
  size_t backtrace_max_size_bytes_ = 0;

  size_t fill_on_alloc_bytes_ = 0;
  size_t fill_on_free_bytes_ = 0;

  size_t expand_alloc_bytes_ = 0;

  size_t free_track_allocations_ = 0;
  size_t free_track_backtrace_num_frames_ = 0;

  int record_allocs_signal_ = 0;
  size_t record_allocs_num_entries_ = 0;
  std::string record_allocs_file_;
  bool record_allocs_on_exit_ = false;

  uint64_t options_ = 0;
  uint8_t fill_alloc_value_;
  uint8_t fill_free_value_;
  uint8_t front_guard_value_;
  uint8_t rear_guard_value_;

  int check_unreachable_signal_ = 0;
  int log_allocator_stats_signal_ = 0;
};
