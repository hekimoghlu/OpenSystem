/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 16, 2021.
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

#include "../../stdexec/concepts.hpp"
#include "../../stdexec/execution.hpp"
#include "../sequence_senders.hpp"

namespace exec {
  namespace __empty_sequence {

    using namespace stdexec;

    template <class _ReceiverId>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using __id = __operation;
        STDEXEC_ATTRIBUTE(no_unique_address) _Receiver __rcvr_;

        void start() & noexcept {
          stdexec::set_value(static_cast<_Receiver&&>(__rcvr_));
        }
      };
    };

    struct __sender {
      struct __t {
        using __id = __sender;
        using sender_concept = sequence_sender_t;
        using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t()>;
        using item_types = exec::item_types<>;

        template <__decays_to<__t> _Self, receiver_of<completion_signatures> _Rcvr>
        STDEXEC_MEMFN_DECL(auto subscribe)(this _Self&&, _Rcvr __rcvr)
          noexcept(__nothrow_move_constructible<_Rcvr>) {
          return stdexec::__t<__operation<stdexec::__id<_Rcvr>>>{static_cast<_Rcvr&&>(__rcvr)};
        }
      };
    };

    struct empty_sequence_t {
      auto operator()() const noexcept -> __t<__sender> {
        return {};
      }
    };

  } // namespace __empty_sequence

  using __empty_sequence::empty_sequence_t;
  inline constexpr empty_sequence_t empty_sequence{};

} // namespace exec
