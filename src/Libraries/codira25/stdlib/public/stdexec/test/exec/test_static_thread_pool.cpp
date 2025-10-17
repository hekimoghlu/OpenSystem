/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 21, 2024.
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

#include "catch2/catch.hpp"
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include <mutex>
#include <thread>
#include <unordered_set>
namespace ex = stdexec;

TEST_CASE(
  "static_thread_pool::get_scheduler_on_thread Test start on a specific thread",
  "[types][static_thread_pool]") {
  constexpr const size_t num_of_threads = 5;
  exec::static_thread_pool pool{num_of_threads};

  std::unordered_set<std::thread::id> thread_ids;
  for (size_t i = 0; i < num_of_threads; ++i) {
    auto sender = ex::schedule(pool.get_scheduler_on_thread(i))
                | ex::then([&]() -> void { thread_ids.insert(std::this_thread::get_id()); });
    ex::sync_wait(std::move(sender));
  }
  REQUIRE(thread_ids.size() == num_of_threads);
}

TEST_CASE(
  "bulk on static_thread_pool executes on multiple threads",
  "[types][static_thread_pool]") {
  constexpr const size_t num_of_threads = 5;
  exec::static_thread_pool pool{num_of_threads};

  std::mutex mtx;
  std::unordered_set<std::thread::id> thread_ids;
  auto sender = ex::starts_on(
    pool.get_scheduler(), ex::just() | ex::bulk(ex::par_unseq, num_of_threads, [&](size_t) -> void {
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                            std::lock_guard lock(mtx);
                            thread_ids.insert(std::this_thread::get_id());
                          }));
  ex::sync_wait(std::move(sender));
  REQUIRE(thread_ids.size() == num_of_threads);
}

TEST_CASE(
  "bulk on static_thread_pool executes on multiple threads, take 2",
  "[types][static_thread_pool]") {
  constexpr const size_t num_of_threads = 5;
  exec::static_thread_pool pool{num_of_threads};

  std::mutex mtx;
  std::unordered_set<std::thread::id> thread_ids;
  auto sender = ex::schedule(pool.get_scheduler())
              | ex::bulk(ex::par_unseq, num_of_threads, [&](size_t) -> void {
                  std::this_thread::sleep_for(std::chrono::milliseconds(100));
                  std::lock_guard lock(mtx);
                  thread_ids.insert(std::this_thread::get_id());
                });
  ex::sync_wait(std::move(sender));
  REQUIRE(thread_ids.size() == num_of_threads);
}
