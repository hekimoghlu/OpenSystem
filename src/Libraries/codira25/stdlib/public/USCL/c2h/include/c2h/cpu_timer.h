/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 28, 2022.
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

#include <uscl/std/tuple>

#include <chrono>
#include <iostream>
#include <string>

// #define C2H_DEBUG_TIMING

#ifdef C2H_DEBUG_TIMING
#  define C2H_TIME_SECTION_INIT()  [[maybe_unused]] c2h::cpu_timer _c2h_timer_
#  define C2H_TIME_SECTION_RESET() _c2h_timer_.reset()
#  define C2H_TIME_SECTION(label)  _c2h_timer_.print_elapsed_seconds_and_reset(label)
#  define C2H_TIME_SCOPE(label)    [[maybe_unused]] c2h::scoped_cpu_timer _c2h_scoped_cpu_timer_(label)
#else
#  define C2H_TIME_SECTION_INIT()  /* no-op */ []() {}()
#  define C2H_TIME_SECTION_RESET() /* no-op */ []() {}()
#  define C2H_TIME_SECTION(label)  /* no-op */ []() {}()
#  define C2H_TIME_SCOPE(label)    /* no-op */ []() {}()
#endif

namespace c2h
{

class cpu_timer
{
  std::chrono::high_resolution_clock::time_point m_start;

public:
  cpu_timer()
      : m_start(std::chrono::high_resolution_clock::now())
  {}

  void reset()
  {
    m_start = std::chrono::high_resolution_clock::now();
  }

  int elapsed_ms() const
  {
    auto duration = std::chrono::high_resolution_clock::now() - m_start;
    auto ms       = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    return static_cast<int>(ms.count());
  }

  std::uint64_t elapsed_us() const
  {
    auto duration = std::chrono::high_resolution_clock::now() - m_start;
    auto us       = std::chrono::duration_cast<std::chrono::microseconds>(duration);
    return static_cast<std::uint64_t>(us.count());
  }

  void print_elapsed_seconds(const std::string& label)
  {
    printf("%0.6f s: %s\n", this->elapsed_us() / 1000000.f, label.c_str());
  }

  void print_elapsed_seconds_and_reset(const std::string& label)
  {
    this->print_elapsed_seconds(label);
    this->reset();
  }
};

class scoped_cpu_timer
{
  cpu_timer m_timer;
  std::string m_label;

public:
  explicit scoped_cpu_timer(std::string label)
      : m_label(std::move(label))
  {}

  ~scoped_cpu_timer()
  {
    m_timer.print_elapsed_seconds(m_label);
  }
};

} // namespace c2h
