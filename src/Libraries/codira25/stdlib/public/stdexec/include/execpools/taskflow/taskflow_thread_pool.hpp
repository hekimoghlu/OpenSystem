/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 13, 2024.
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

#include <taskflow/taskflow.hpp>

#include <execpools/thread_pool_base.hpp>

namespace execpools {

  class taskflow_thread_pool : public execpools::thread_pool_base<taskflow_thread_pool> {
   public:
    //! Constructor forwards to tbb::task_arena constructor:
    template <class... Args>
      requires stdexec::constructible_from<tf::Executor, Args...>
    explicit taskflow_thread_pool(Args&&... args)
      : executor_(std::forward<Args>(args)...) {
    }

    [[nodiscard]]
    auto available_parallelism() const -> std::uint32_t {
      return static_cast<std::uint32_t>(executor_.num_workers());
    }
   private:
    [[nodiscard]]
    static constexpr auto forward_progress_guarantee() -> stdexec::forward_progress_guarantee {
      return stdexec::forward_progress_guarantee::parallel;
    }

    friend execpools::thread_pool_base<taskflow_thread_pool>;

    template <class PoolType, class ReceiverId>
    friend struct execpools::operation;

    void enqueue(execpools::task_base* task, std::uint32_t tid = 0) noexcept {
      executor_.silent_async([task, tid] { task->__execute(task, /*tid=*/tid); });
    }

    tf::Executor executor_;
  };
} // namespace execpools
