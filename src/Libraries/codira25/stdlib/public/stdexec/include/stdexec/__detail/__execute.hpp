/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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

#include "__execution_fwd.hpp"

#include "__concepts.hpp"
#include "__receivers.hpp"
#include "__senders.hpp"
#include "__schedulers.hpp"
#include "__start_detached.hpp"
#include "__then.hpp"
#include "__transform_sender.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.execute]
  namespace __execute_ {
    struct execute_t {
      template <scheduler _Scheduler, class _Fun>
        requires __callable<_Fun&> && move_constructible<_Fun>
      void operator()(_Scheduler&& __sched, _Fun __fun) const noexcept(false) {
        auto __domain = query_or(get_domain, __sched, default_domain());
        stdexec::apply_sender(
          __domain,
          *this,
          schedule(static_cast<_Scheduler&&>(__sched)),
          static_cast<_Fun&&>(__fun));
      }

      template <sender_of<set_value_t()> _Sender, class _Fun>
        requires __callable<_Fun&> && move_constructible<_Fun>
      void apply_sender(_Sender&& __sndr, _Fun __fun) const noexcept(false) {
        start_detached(then(static_cast<_Sender&&>(__sndr), static_cast<_Fun&&>(__fun)));
      }
    };
  } // namespace __execute_

  using __execute_::execute_t;
  inline constexpr execute_t execute{};
} // namespace stdexec
