/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 15, 2024.
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

#include "nvexec/stream_context.cuh"
#include "common.cuh"

namespace ex = stdexec;

using nvexec::is_on_gpu;

namespace {

  TEST_CASE("nvexec split returns a sender", "[cuda][stream][adaptors][split]") {
    nvexec::stream_context stream_ctx{};
    auto snd = ex::split(ex::schedule(stream_ctx.get_scheduler()));
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("nvexec split works", "[cuda][stream][adaptors][split]") {
    nvexec::stream_context stream_ctx{};

    auto fork = ex::schedule(stream_ctx.get_scheduler()) | ex::then([=] { return is_on_gpu(); })
              | ex::split();

    auto b1 = fork | ex::then([](bool on_gpu) { return on_gpu * 24; });
    auto b2 = fork | ex::then([](bool on_gpu) { return on_gpu * 42; });

    auto [v1] = stdexec::sync_wait(std::move(b1)).value();
    auto [v2] = stdexec::sync_wait(std::move(b2)).value();

    REQUIRE(v1 == 24);
    REQUIRE(v2 == 42);
  }

  TEST_CASE("nvexec split can preceed a sender without values", "[cuda][stream][adaptors][split]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::schedule(stream_ctx.get_scheduler()) | ex::split() | a_sender([=]() noexcept {
                 if (is_on_gpu()) {
                   flags.set();
                 }
               });

    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec split can succeed a sender", "[cuda][stream][adaptors][split]") {
    SECTION("without values") {
      nvexec::stream_context stream_ctx{};
      flags_storage_t<2> flags_storage{};
      auto flags = flags_storage.get();

      auto snd = ex::schedule(stream_ctx.get_scheduler()) | a_sender([flags] {
                   if (is_on_gpu()) {
                     flags.set(1);
                   }
                 })
               | ex::split() | ex::then([flags] {
                   if (is_on_gpu()) {
                     flags.set(0);
                   }
                 });
      stdexec::sync_wait(std::move(snd));

      REQUIRE(flags_storage.all_set_once());
    }

    SECTION("with values") {
      nvexec::stream_context stream_ctx{};
      flags_storage_t flags_storage{};
      auto flags = flags_storage.get();

      auto snd = ex::schedule(stream_ctx.get_scheduler())
               | a_sender([]() -> bool { return is_on_gpu(); }) | ex::split()
               | ex::then([flags](bool a_sender_was_on_gpu) {
                   if (a_sender_was_on_gpu && is_on_gpu()) {
                     flags.set();
                   }
                 });
      stdexec::sync_wait(std::move(snd)).value();

      REQUIRE(flags_storage.all_set_once());
    }
  }
} // namespace
