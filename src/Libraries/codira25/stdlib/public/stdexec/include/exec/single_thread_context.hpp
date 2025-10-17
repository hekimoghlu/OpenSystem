/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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

#include "../stdexec/execution.hpp"

#include <thread>

namespace exec {
  class single_thread_context {
    stdexec::run_loop loop_;
    std::thread thread_;

   public:
    single_thread_context()
      : loop_()
      , thread_([this] { loop_.run(); }) {
    }

    ~single_thread_context() {
      loop_.finish();
      thread_.join();
    }

    auto get_scheduler() noexcept {
      return loop_.get_scheduler();
    }

    [[nodiscard]]
    auto get_thread_id() const noexcept -> std::thread::id {
      return thread_.get_id();
    }
  };
} // namespace exec
