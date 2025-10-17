/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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
#pragma once

#include <type_traits>
#include <utility>
#include <asioexec/asio_config.hpp>
#include <asioexec/executor_with_default.hpp>

namespace asioexec {

  template <typename CompletionToken, typename IoObject>
  using as_default_on_t =
    typename std::remove_cvref_t<IoObject>::template rebind_executor<executor_with_default<
      std::remove_cvref_t<decltype(std::declval<IoObject&>().get_executor())>,
      CompletionToken>>::other;

  namespace detail::as_default_on {

    template <typename CompletionToken>
    struct t {
      template <typename IoObject>
      constexpr asioexec::as_default_on_t<CompletionToken, IoObject>
        operator()(IoObject&& io) const {
        return asioexec::as_default_on_t<CompletionToken, IoObject>((IoObject&&) io);
      }
    };

  } // namespace detail::as_default_on

  template <typename CompletionToken>
  inline constexpr detail::as_default_on::t<CompletionToken> as_default_on;

} // namespace asioexec
