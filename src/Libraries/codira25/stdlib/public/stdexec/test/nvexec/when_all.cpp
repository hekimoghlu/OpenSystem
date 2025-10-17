/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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
#include <stdexec/execution.hpp>

#include "nvexec/stream/common.cuh"
#include "nvexec/stream_context.cuh"
#include "common.cuh"

namespace ex = stdexec;

using nvexec::is_on_gpu;

namespace {

  TEST_CASE("nvexec when_all returns a sender", "[cuda][stream][adaptors][when_all]") {
    nvexec::stream_context stream_ctx{};
    auto snd = ex::when_all(
      ex::schedule(stream_ctx.get_scheduler()), ex::schedule(stream_ctx.get_scheduler()));
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("nvexec when_all works", "[cuda][stream][adaptors][when_all]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<2> flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::when_all(
      ex::schedule(stream_ctx.get_scheduler()) | ex::then([=] {
        if (is_on_gpu()) {
          flags.set(0);
        }
      }),
      ex::schedule(stream_ctx.get_scheduler()) | ex::then([=] {
        if (is_on_gpu()) {
          flags.set(1);
        }
      }));
    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec when_all returns values", "[cuda][stream][adaptors][when_all]") {
    nvexec::stream_context stream_ctx{};

    auto snd = ex::when_all(
      ex::schedule(stream_ctx.get_scheduler()) | ex::then([] { return is_on_gpu() * 24; }),
      ex::schedule(stream_ctx.get_scheduler()) | ex::then([] { return is_on_gpu() * 42; }));
    auto [v1, v2] = stdexec::sync_wait(std::move(snd)).value();

    REQUIRE(v1 == 24);
    REQUIRE(v2 == 42);
  }

  TEST_CASE("nvexec when_all with many senders", "[cuda][stream][adaptors][when_all]") {
    nvexec::stream_context stream_ctx{};

    auto snd = ex::when_all(
      ex::schedule(stream_ctx.get_scheduler()) | ex::then([] { return is_on_gpu() * 1; }),
      ex::schedule(stream_ctx.get_scheduler()) | ex::then([] { return is_on_gpu() * 2; }),
      ex::schedule(stream_ctx.get_scheduler()) | ex::then([] { return is_on_gpu() * 3; }),
      ex::schedule(stream_ctx.get_scheduler()) | ex::then([] { return is_on_gpu() * 4; }),
      ex::schedule(stream_ctx.get_scheduler()) | ex::then([] { return is_on_gpu() * 5; }));
    auto [v1, v2, v3, v4, v5] = stdexec::sync_wait(std::move(snd)).value();

    REQUIRE(v1 == 1);
    REQUIRE(v2 == 2);
    REQUIRE(v3 == 3);
    REQUIRE(v4 == 4);
    REQUIRE(v5 == 5);
  }

  TEST_CASE("nvexec when_all works with unknown senders", "[cuda][stream][adaptors][when_all]") {
    nvexec::stream_context stream_ctx{};
    auto sch = stream_ctx.get_scheduler();

    auto snd = ex::when_all(
      ex::schedule(sch) | ex::then([]() -> int { return is_on_gpu() * 24; }),
      ex::schedule(sch) | a_sender([]() -> int { return is_on_gpu() * 42; }));
    auto [v1, v2] = stdexec::sync_wait(std::move(snd)).value();

    REQUIRE(v1 == 24);
    REQUIRE(v2 == 42);
  }
} // namespace
