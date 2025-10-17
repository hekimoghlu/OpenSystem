/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 6, 2023.
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
#include "exec/into_tuple.hpp"
#include "test_common/senders.hpp"
#include "test_common/type_helpers.hpp"

#include <catch2/catch.hpp>

using namespace stdexec;

namespace {

  TEST_CASE("into_tuple is a sender", "[adaptors][into_tuple]") {
    auto s = exec::into_tuple(just(42, 56));
    STATIC_REQUIRE(sender<decltype(s)>);
  }

  TEST_CASE("into_tuple is a sender in empty env", "[adaptors][into_tuple]") {
    using void_sender = result_of_t<exec::into_tuple, a_sender_of<set_value_t()>>;
    STATIC_REQUIRE(sender_in<void_sender, ex::env<>>);
    STATIC_REQUIRE(
      set_equivalent<
        completion_signatures_of_t<void_sender, ex::env<>>,
        completion_signatures<set_error_t(std::exception_ptr), set_value_t(std::tuple<>)>
      >);

    using ints_sender = result_of_t<exec::into_tuple, a_sender_of<set_value_t(int&, int)>>;
    STATIC_REQUIRE(sender_in<ints_sender, ex::env<>>);
    STATIC_REQUIRE(
      set_equivalent<
        completion_signatures_of_t<ints_sender, ex::env<>>,
        completion_signatures<set_error_t(std::exception_ptr), set_value_t(std::tuple<int, int>)>
      >);
  }

  TEST_CASE(
    "into_tuple rejects senders unless they have a single value completion",
    "[adaptors][into_tuple]") {
    using error_sender =
      result_of_t<exec::into_tuple, a_sender_of<set_error_t(std::exception_ptr)>>;
    STATIC_REQUIRE_FALSE(sender_in<error_sender, ex::env<>>);

    using stopped_sender = result_of_t<exec::into_tuple, a_sender_of<set_stopped_t()>>;
    STATIC_REQUIRE_FALSE(sender_in<stopped_sender, ex::env<>>);

    using multi_value_sender =
      result_of_t<exec::into_tuple, a_sender_of<set_value_t(), set_value_t(int)>>;
    STATIC_REQUIRE_FALSE(sender_in<multi_value_sender, ex::env<>>);
  }

  TEST_CASE("trivial into_tuple example works", "[adaptors][into_tuple]") {
    auto s = exec::into_tuple(just(42, 56));
    auto [i, j] = get<0>(sync_wait(s).value());
    CHECK(i == 42);
    CHECK(j == 56);
  }
} // namespace
