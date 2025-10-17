/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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
#include "exec/static_thread_pool.hpp"

namespace ex = stdexec;
using exec::async_scope;
using stdexec::sync_wait;

namespace {

  TEST_CASE("async_scope can be created and them immediately destructed", "[async_scope][dtor]") {
    async_scope scope;
    (void) scope;
  }

  TEST_CASE("async_scope destruction after spawning work into it", "[async_scope][dtor]") {
    exec::static_thread_pool pool{4};
    ex::scheduler auto sch = pool.get_scheduler();
    std::atomic<int> counter{0};
    {
      async_scope scope;

      // Add some work into the scope
      for (int i = 0; i < 10; i++)
        scope.spawn(ex::starts_on(sch, ex::just() | ex::then([&] { counter++; })));

      // Wait on the work, before calling destructor
      sync_wait(scope.on_empty());
    }
    // We should have all the work executed
    REQUIRE(counter == 10);
  }
} // namespace
