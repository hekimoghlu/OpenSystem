/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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

#include "../../relacy/relacy_std.hpp"
#include "../../relacy/relacy_cli.hpp"

#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>

using rl::nvar;
using rl::nvolatile;
using rl::mutex;

namespace ex = stdexec;

struct split_bug : rl::test_suite<split_bug, 1> {
  static size_t const dynamic_thread_count = 2;

  void thread(unsigned) {
    exec::static_thread_pool pool{1};
    auto split = ex::schedule(pool.get_scheduler()) | ex::then([] { return 42; }) | ex::split();

    auto [val] = ex::sync_wait(split).value();
    RL_ASSERT(val == 42);
  }
};

auto main() -> int {
  rl::test_params p;
  p.iteration_count = 50000;
  p.execution_depth_limit = 10000;
  rl::simulate<split_bug>(p);
  return 0;
}
