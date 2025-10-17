/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 18, 2024.
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

STDEXEC_PRAGMA_IGNORE_GNU("-Wdeprecated-declarations")
STDEXEC_PRAGMA_IGNORE_EDG(deprecated_entity_with_custom_message)
STDEXEC_PRAGMA_IGNORE_MSVC(4996)

namespace ex = stdexec;

namespace {

  struct my_forwarding_query_t {
    [[nodiscard]]
    constexpr auto query(ex::forwarding_query_t) const noexcept -> bool {
      return true;
    }
  };

  inline constexpr my_forwarding_query_t my_forwarding_query{};

  struct my_derived_forwarding_query_t : ex::forwarding_query_t { };

  inline constexpr my_derived_forwarding_query_t my_derived_forwarding_query{};

  struct my_non_forwarding_query_t { };

  inline constexpr my_non_forwarding_query_t my_non_forwarding_query{};

  TEST_CASE("exec.queries are forwarding queries", "[exec.queries][forwarding_queries]") {
    static_assert(ex::forwarding_query(ex::get_allocator));
    static_assert(ex::forwarding_query(ex::get_stop_token));
    static_assert(ex::forwarding_query(ex::get_scheduler));
    static_assert(ex::forwarding_query(ex::get_delegation_scheduler));
    static_assert(
      std::is_same_v<ex::get_delegatee_scheduler_t, ex::get_delegation_scheduler_t>,
      "Deprecated type is the same type.");
    static_assert(&ex::get_delegatee_scheduler == &ex::get_delegation_scheduler);
    static_assert(ex::forwarding_query(ex::get_completion_scheduler<ex::set_value_t>));
    static_assert(ex::forwarding_query(ex::get_completion_scheduler<ex::set_error_t>));
    static_assert(ex::forwarding_query(ex::get_completion_scheduler<ex::set_stopped_t>));

    static_assert(ex::forwarding_query(my_forwarding_query));
    static_assert(ex::forwarding_query(my_derived_forwarding_query));
    static_assert(!ex::forwarding_query(my_non_forwarding_query));
  }
} // namespace
