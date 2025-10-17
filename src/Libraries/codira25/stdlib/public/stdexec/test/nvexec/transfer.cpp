/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 1, 2024.
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

#include "exec/inline_scheduler.hpp"
#include "nvexec/stream_context.cuh"
#include "common.cuh"

namespace ex = stdexec;

using nvexec::is_on_gpu;

namespace {

  TEST_CASE(
    "nvexec transfer to stream context returns a sender",
    "[cuda][stream][adaptors][transfer]") {
    nvexec::stream_context stream_ctx{};
    stdexec::inline_scheduler cpu{};
    nvexec::stream_scheduler gpu = stream_ctx.get_scheduler();

    auto snd = ex::schedule(cpu) | ex::continues_on(gpu);
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE(
    "nvexec transfer from stream context returns a sender",
    "[cuda][stream][adaptors][transfer]") {
    nvexec::stream_context stream_ctx{};

    stdexec::inline_scheduler cpu{};
    nvexec::stream_scheduler gpu = stream_ctx.get_scheduler();

    auto snd = ex::schedule(gpu) | ex::continues_on(cpu);
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("nvexec transfer changes context to GPU", "[cuda][stream][adaptors][transfer]") {
    nvexec::stream_context stream_ctx{};

    stdexec::inline_scheduler cpu{};
    nvexec::stream_scheduler gpu = stream_ctx.get_scheduler();

    auto snd = ex::schedule(cpu) | ex::then([=] {
                 if (!is_on_gpu()) {
                   return 1;
                 }
                 return 0;
               })
             | ex::continues_on(gpu) | ex::then([=](int val) -> int {
                 if (is_on_gpu() && val == 1) {
                   return 2;
                 }
                 return 0;
               });
    const auto [result] = stdexec::sync_wait(std::move(snd)).value();

    REQUIRE(result == 2);
  }

  TEST_CASE("nvexec transfer changes context from GPU", "[cuda][stream][adaptors][transfer]") {
    nvexec::stream_context stream_ctx{};

    stdexec::inline_scheduler cpu{};
    nvexec::stream_scheduler gpu = stream_ctx.get_scheduler();

    auto snd = ex::schedule(gpu) | ex::then([=] {
                 if (is_on_gpu()) {
                   return 1;
                 }
                 return 0;
               })
             | ex::continues_on(cpu) | ex::then([=](int val) -> int {
                 if (!is_on_gpu() && val == 1) {
                   return 2;
                 }
                 return 0;
               });
    const auto [result] = stdexec::sync_wait(std::move(snd)).value();

    REQUIRE(result == 2);
  }

  TEST_CASE("nvexec transfer_just changes context to GPU", "[cuda][stream][adaptors][transfer]") {
    nvexec::stream_context stream_ctx{};
    nvexec::stream_scheduler gpu = stream_ctx.get_scheduler();

    auto snd = ex::transfer_just(gpu, 42) | ex::then([=](auto i) {
                 if (is_on_gpu() && i == 42) {
                   return true;
                 }
                 return false;
               });
    const auto [result] = stdexec::sync_wait(std::move(snd)).value();

    REQUIRE(result == true);
  }

  TEST_CASE("nvexec transfer_just supports move-only types", "[cuda][stream][adaptors][transfer]") {
    nvexec::stream_context stream_ctx{};
    nvexec::stream_scheduler gpu = stream_ctx.get_scheduler();

    auto snd = ex::transfer_just(gpu, move_only_t{42}) | ex::then([=](move_only_t&& val) noexcept {
                 if (is_on_gpu() && val.contains(42)) {
                   return true;
                 }
                 return false;
               });
    const auto [result] = stdexec::sync_wait(std::move(snd)).value();

    REQUIRE(result == true);
  }

  TEST_CASE("nvexec transfer supports move-only types", "[cuda][stream][adaptors][transfer]") {
    nvexec::stream_context stream_ctx{};

    stdexec::inline_scheduler cpu{};
    nvexec::stream_scheduler gpu = stream_ctx.get_scheduler();

    auto snd = ex::schedule(gpu) | ex::then([] { return move_only_t{42}; }) | ex::continues_on(cpu)
             | ex::then([=](move_only_t val) noexcept {
                 if (!is_on_gpu() && val.contains(42)) {
                   return true;
                 }
                 return false;
               });
    const auto [result] = stdexec::sync_wait(std::move(snd)).value();

    REQUIRE(result == true);
  }
} // namespace
