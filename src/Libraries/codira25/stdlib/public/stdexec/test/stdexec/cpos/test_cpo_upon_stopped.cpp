/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 4, 2022.
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
#include "cpo_helpers.cuh"
#include <catch2/catch.hpp>

namespace {

  TEST_CASE("upon_stopped is customizable", "[cpo][cpo_upon_stopped]") {
    const auto f = []() {
    };

    SECTION("by free standing sender") {
      cpo_test_sender_t<ex::upon_stopped_t> snd{};

      {
        constexpr scope_t scope = decltype(snd | ex::upon_stopped(f))::scope;
        STATIC_REQUIRE(scope == scope_t::free_standing);
      }

      {
        constexpr scope_t scope = decltype(ex::upon_stopped(snd, f))::scope;
        STATIC_REQUIRE(scope == scope_t::free_standing);
      }
    }
  }
} // namespace
