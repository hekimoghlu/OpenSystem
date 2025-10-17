/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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
#include <exec/env.hpp>

#include "nvexec/stream_context.cuh"
#include "common.cuh"

namespace ex = stdexec;

using nvexec::is_on_gpu;

namespace {

  TEST_CASE("nvexec let_error returns a sender", "[cuda][stream][adaptors][let_error]") {
    nvexec::stream_context stream_ctx{};

    auto snd = ex::just_error(42) | ex::continues_on(stream_ctx.get_scheduler())
             | ex::let_error([](int) { return ex::just(); });
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("nvexec let_error executes on GPU", "[cuda][stream][adaptors][let_error]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::just_error(42) | ex::continues_on(stream_ctx.get_scheduler())
             | ex::let_error([=](int err) {
                 if (is_on_gpu() && err == 42) {
                   flags.set();
                 }

                 return ex::just()
                      | exec::write_attrs(ex::prop{ex::get_domain, nvexec::stream_domain()});
               });
    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE(
    "nvexec let_error can preceed a sender without values",
    "[cuda][stream][adaptors][let_error]") {
    nvexec::stream_context stream_ctx{};

    flags_storage_t<2> flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::just_error(42) | ex::continues_on(stream_ctx.get_scheduler())
             | ex::let_error([flags](int err) {
                 if (is_on_gpu() && err == 42) {
                   flags.set(0);
                 }

                 return ex::just()
                      | exec::write_attrs(ex::prop{ex::get_domain, nvexec::stream_domain()});
               })
             | a_sender([flags] {
                 if (is_on_gpu()) {
                   flags.set(1);
                 }
               });
    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }

  TEST_CASE("nvexec let_error can succeed a sender", "[cuda][stream][adaptors][let_error]") {
    nvexec::stream_context stream_ctx{};
    nvexec::stream_scheduler sch = stream_ctx.get_scheduler();
    flags_storage_t flags_storage{};
    auto flags = flags_storage.get();

    auto snd = ex::just_error(42) | ex::continues_on(stream_ctx.get_scheduler())
             | a_sender([]() noexcept {}) | ex::let_error([=](int err) {
                 if (is_on_gpu() && err == 42) {
                   flags.set();
                 }

                 return ex::schedule(sch);
               });
    stdexec::sync_wait(std::move(snd));

    REQUIRE(flags_storage.all_set_once());
  }
} // namespace
