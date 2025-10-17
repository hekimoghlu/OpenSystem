/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 23, 2024.
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

namespace ex = stdexec;

namespace {

  STDEXEC_PRAGMA_PUSH()
  STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-function")
  STDEXEC_PRAGMA_IGNORE_GNU("-Wunneeded-internal-declaration")

  struct op_except {
    op_except() = default;
    op_except(op_except&&) = delete;

    void start() & {
    }
  };

  struct op_noexcept {
    op_noexcept() = default;
    op_noexcept(op_noexcept&&) = delete;

    void start() & noexcept {
    }
  };

  STDEXEC_PRAGMA_POP()

  // TEST_CASE(
  //   "type with start CPO that throws is not an operation_state",
  //   "[concepts][operation_state]") {
  //   REQUIRE(!ex::operation_state<op_except>);
  // }

  TEST_CASE("type with start CPO noexcept is an operation_state", "[concepts][operation_state]") {
    REQUIRE(ex::operation_state<op_noexcept>);
  }

  TEST_CASE("reference type is not an operation_state", "[concepts][operation_state]") {
    REQUIRE(!ex::operation_state<op_noexcept&>);
  }

  TEST_CASE("pointer type is not an operation_state", "[concepts][operation_state]") {
    REQUIRE(!ex::operation_state<op_noexcept*>);
  }
} // namespace
