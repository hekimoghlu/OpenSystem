/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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

  TEST_CASE("nvexec let_value returns a sender", "[cuda][stream][adaptors][let_value]") {
    nvexec::stream_context stream_ctx{};
    auto snd = ex::let_value(ex::schedule(stream_ctx.get_scheduler()), [] { return ex::just(); });
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("nvexec let_value executes on GPU", "[cuda][stream][adaptors][let_value]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::schedule(stream_ctx.get_scheduler()) | ex::let_value([=] {
                 if (is_on_gpu()) {
                   flags.set();
                 }
                 return ex::just();
               });
    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec let_value accepts values on GPU", "[cuda][stream][adaptors][let_value]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::schedule(stream_ctx.get_scheduler()) | ex::then([]() -> int { return 42; })
             | ex::let_value([=](int val) {
                 if (is_on_gpu()) {
                   if (val == 42) {
                     flags.set();
                   }
                 }
                 return ex::just();
               });
    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE(
    "nvexec let_value accepts multiple values on GPU",
    "[cuda][stream][adaptors][let_value]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::transfer_just(stream_ctx.get_scheduler(), 42, 4.2)
             | ex::let_value([=](int i, double d) {
                 if (is_on_gpu()) {
                   if (i == 42 && d == 4.2) {
                     flags.set();
                   }
                 }
                 return ex::just();
               });
    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec let_value returns values on GPU", "[cuda][stream][adaptors][let_value]") {
    nvexec::stream_context stream_ctx{};

    auto snd = ex::schedule(stream_ctx.get_scheduler())
             | ex::let_value([=]() { return ex::just(is_on_gpu()); });
    const auto [result] = stdexec::sync_wait(std::move(snd)).value();

    REQUIRE(result == 1);
  }

  TEST_CASE(
    "nvexec let_value can preceed a sender without values",
    "[cuda][stream][adaptors][let_value]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<2> flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::schedule(stream_ctx.get_scheduler()) | ex::let_value([flags] {
                 if (is_on_gpu()) {
                   flags.set(0);
                 }

                 return ex::just();
               })
             | a_sender([flags] {
                 if (is_on_gpu()) {
                   flags.set(1);
                 }
               });
    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec let_value can succeed a sender", "[cuda][stream][adaptors][let_value]") {
    nvexec::stream_context stream_ctx{};
    nvexec::stream_scheduler sch = stream_ctx.get_scheduler();
    flags_storage_t flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::schedule(sch) | a_sender([]() noexcept {}) | ex::let_value([=] {
                 if (is_on_gpu()) {
                   flags.set();
                 }

                 return ex::schedule(sch);
               });
    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec let_value can read a property", "[cuda][stream][adaptors][let_value]") {
    nvexec::stream_context stream_ctx{};
    nvexec::stream_scheduler sch = stream_ctx.get_scheduler();
    flags_storage_t flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::schedule(sch) | ex::let_value([] { return nvexec::get_stream(); })
             | ex::then([flags](cudaStream_t stream) {
                 if (is_on_gpu()) {
                   flags.set();
                 }
                 return stream;
               });
    auto [stream] = stdexec::sync_wait(std::move(snd)).value();
    static_assert(ex::same_as<decltype(+stream), cudaStream_t>);

    REQUIRE(flags_storage.all_set_once());
  }
} // namespace
