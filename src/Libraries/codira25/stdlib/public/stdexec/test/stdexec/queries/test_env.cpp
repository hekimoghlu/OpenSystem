/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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

namespace {

  template <typename T>
  concept can_get_domain = requires(const T& t) { t.query(::stdexec::get_domain); };

  namespace zero {

    using env = ::stdexec::env<>;
    static_assert(std::is_same_v<::stdexec::never_stop_token, ::stdexec::stop_token_of_t<env>>);
    static_assert(!can_get_domain<env>);

  } // namespace zero

  namespace one {
    using env = ::stdexec::env<::stdexec::env<>>;
    static_assert(std::is_same_v<::stdexec::never_stop_token, ::stdexec::stop_token_of_t<env>>);
    static_assert(!can_get_domain<env>);
  } // namespace one

  namespace two {
    using env = ::stdexec::env<::stdexec::env<>, ::stdexec::env<>>;
    static_assert(std::is_same_v<::stdexec::never_stop_token, ::stdexec::stop_token_of_t<env>>);
    static_assert(!can_get_domain<env>);
  } // namespace two

  namespace three {
    using env = ::stdexec::env<::stdexec::env<>, ::stdexec::env<>, ::stdexec::env<>>;
    static_assert(std::is_same_v<::stdexec::never_stop_token, ::stdexec::stop_token_of_t<env>>);
    static_assert(!can_get_domain<env>);
  } // namespace three
} // namespace
