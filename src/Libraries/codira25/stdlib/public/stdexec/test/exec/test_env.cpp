/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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

namespace {
  // Two dummy properties:
  constexpr struct Foo
    : stdexec::__query<Foo>
    , stdexec::forwarding_query_t {
    using stdexec::__query<Foo>::operator();
  } foo{};

  constexpr struct Bar : stdexec::__query<Bar> {
    static constexpr auto query(stdexec::forwarding_query_t) noexcept -> bool {
      return true;
    }
  } bar{};

  TEST_CASE("Test make_env works", "[env]") {
    auto e = stdexec::prop{foo, 42};
    CHECK(foo(e) == 42);

    auto e2 = exec::make_env(e, stdexec::prop{bar, 43});
    CHECK(foo(e2) == 42);
    CHECK(bar(e2) == 43);

    auto e3 = exec::make_env(e2, stdexec::prop{foo, 44});
    CHECK(foo(e3) == 44);
    CHECK(bar(e3) == 43);

    auto e4 = exec::without(e3, foo);
    STATIC_REQUIRE(!std::invocable<Foo, decltype(e4)>);
    CHECK(bar(e4) == 43);
  }
} // namespace
