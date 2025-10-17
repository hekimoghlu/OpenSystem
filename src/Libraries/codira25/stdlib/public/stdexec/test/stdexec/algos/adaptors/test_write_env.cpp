/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 30, 2021.
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

#include <utility>
#include <type_traits>

#include <stdexec/execution.hpp>
#include <test_common/receivers.hpp>

namespace {

  template <typename T>
  struct receiver : expect_void_receiver<> {
    constexpr ::stdexec::env<> get_env() const noexcept {
      return state_->get_env();
    }
    T* state_;
  };

  struct state;

  static_assert(!std::is_same_v<
                void,
                decltype(::stdexec::connect(
                  ::stdexec::just()
                    | ::stdexec::write_env(::stdexec::prop{
                      ::stdexec::get_stop_token,
                      std::declval<::stdexec::inplace_stop_source&>().get_token()}),
                  receiver<state>{{}, nullptr}))>);

  struct state {
    constexpr ::stdexec::env<> get_env() const noexcept {
      return {};
    }
  };

  TEST_CASE(
    "write_env works when the actual environment is sourced from a type which was initially "
    "incomplete but has since been completed",
    "[adaptors][write_env]") {
    ::stdexec::inplace_stop_source source;
    state s;
    auto op = ::stdexec::connect(
      ::stdexec::just()
        | ::stdexec::write_env(::stdexec::prop{::stdexec::get_stop_token, source.get_token()}),
      receiver<state>{{}, &s});
    ::stdexec::start(op);
  }
} // namespace
