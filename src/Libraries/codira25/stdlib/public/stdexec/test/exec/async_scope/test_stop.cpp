/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 22, 2024.
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
#include "test_common/schedulers.hpp"
#include "test_common/receivers.hpp"
#include "exec/single_thread_context.hpp"

namespace ex = stdexec;
using exec::async_scope;
using stdexec::sync_wait;

namespace {

  TEST_CASE("calling request_stop will be visible in stop_source", "[async_scope][stop]") {
    async_scope scope;

    scope.request_stop();
    REQUIRE(scope.get_stop_source().stop_requested());
  }

  TEST_CASE("calling request_stop will be visible in stop_token", "[async_scope][stop]") {
    async_scope scope;

    scope.request_stop();
    REQUIRE(scope.get_stop_token().stop_requested());
  }

  TEST_CASE(
    "cancelling the associated stop_source will cancel the async_scope object",
    "[async_scope][stop]") {
    bool empty = false;

    {
      impulse_scheduler sch;
      async_scope scope;
      bool called = false;

      // put work in the scope
      scope.spawn(ex::starts_on(sch, ex::just()) | ex::upon_stopped([&] { called = true; }));
      REQUIRE_FALSE(called);

      // start a thread waiting on when the scope is empty:
      exec::single_thread_context thread;
      auto thread_sch = thread.get_scheduler();
      ex::start_detached(
        ex::starts_on(thread_sch, scope.on_empty()) | ex::then([&] { empty = true; }));
      REQUIRE_FALSE(empty);

      // request the scope stop
      scope.get_stop_source().request_stop();

      // execute the work in the scope
      sch.start_next();

      // Should have completed with a stopped signal
      REQUIRE(called);
    } // blocks until the separate thread is joined

    REQUIRE(empty);
  }

  TEST_CASE(
    "cancelling the associated stop_source will be visible in stop_token",
    "[async_scope][stop]") {
    async_scope scope;

    scope.get_stop_source().request_stop();
    REQUIRE(scope.get_stop_token().stop_requested());
  }
} // namespace
