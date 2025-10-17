/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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
#include <exec/variant_sender.hpp>

#include <catch2/catch.hpp>

using namespace stdexec;
using namespace exec;

namespace {

  template <class... Ts>
  struct overloaded : Ts... {
    using Ts::operator()...;
  };
  template <class... Ts>
  overloaded(Ts...) -> overloaded<Ts...>;

  using just_int_t = decltype(just(0));
  using just_void_t = decltype(just());

  TEST_CASE("variant_sender - default constructible", "[types][variant_sender]") {
    variant_sender<just_void_t, just_int_t> variant{just()};
    CHECK(variant.index() == 0);
  }

  TEST_CASE("variant_sender - using an overloaded then adaptor", "[types][variant_sender]") {
    variant_sender<just_void_t, just_int_t> variant = just();
    int index = -1;
    STATIC_REQUIRE(sender<variant_sender<just_void_t, just_int_t>>);
    sync_wait(variant | then([&index](auto... xs) { index = sizeof...(xs); }));
    CHECK(index == 0);

    variant.emplace<1>(just(42));
    auto [value] = sync_wait(
                     variant
                     | then(
                       overloaded{
                         [&index] {
                           index = 0;
                           return 0;
                         },
                         [&index](int xs) {
                           index = 1;
                           return xs;
                         }}))
                     .value();
    CHECK(index == 1);
    CHECK(value == 42);
  }
} // namespace
