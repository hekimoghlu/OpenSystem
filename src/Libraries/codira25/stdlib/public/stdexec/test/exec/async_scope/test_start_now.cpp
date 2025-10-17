/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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

#include <catch2/catch.hpp>
#include <exec/async_scope.hpp>
#include <exec/start_now.hpp>
#include <exec/static_thread_pool.hpp>

#include "test_common/schedulers.hpp"
#include "test_common/receivers.hpp"
#include "test_common/type_helpers.hpp"

namespace ex = stdexec;
using exec::async_scope;
using exec::start_now;
using stdexec::sync_wait;

namespace {

  TEST_CASE("start_now one", "[async_scope][start_now]") {
    bool executed{false};
    async_scope scope;

    // This will be a blocking call
    auto stg = start_now(scope, ex::just() | ex::then([&]() noexcept { executed = true; }));
    sync_wait(stg.async_wait());
    REQUIRE(executed);
  }

  TEST_CASE("start_now two", "[async_scope][start_now]") {
    bool executedA{false};
    bool executedB{false};
    async_scope scope;

    // This will be a blocking call
    auto stg = start_now(
      scope,
      ex::just() | ex::then([&]() noexcept { executedA = true; }),
      ex::just() | ex::then([&]() noexcept { executedB = true; }));
    sync_wait(stg.async_wait());
    REQUIRE(executedA);
    REQUIRE(executedB);
  }

  TEST_CASE("start_now two __root_env", "[async_scope][start_now]") {
    bool executedA{false};
    bool executedB{false};
    async_scope scope;

    // This will be a blocking call
    auto stg = start_now(
      stdexec::__root_env{},
      scope,
      ex::just() | ex::then([&]() noexcept { executedA = true; }),
      ex::just() | ex::then([&]() noexcept { executedB = true; }));
    sync_wait(stg.async_wait());
    REQUIRE(executedA);
    REQUIRE(executedB);
  }

  TEST_CASE("start_now two on pool", "[async_scope][start_now]") {
    bool executedA{false};
    bool executedB{false};
    async_scope scope;
    exec::static_thread_pool pool{2};

    // This will be a blocking call
    auto stg = start_now(
      scope,
      ex::schedule(pool.get_scheduler()) | ex::then([&]() noexcept { executedA = true; }),
      ex::schedule(pool.get_scheduler()) | ex::then([&]() noexcept { executedB = true; }));
    sync_wait(ex::starts_on(pool.get_scheduler(), stg.async_wait()));
    REQUIRE(executedA);
    REQUIRE(executedB);
  }

} // namespace
