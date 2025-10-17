/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 2, 2022.
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

//===--- chrono_utils.h - Utility functions for duration ------ -*- C++ -*-===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//
//
// Specifically, we want ceil() for these types, but that's only available in
// C++17, and we need to build with C++14, so... include a version of the
// necesary code here.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_THREADING_IMPL_CHRONO_UTILS_H
#define LANGUAGE_THREADING_IMPL_CHRONO_UTILS_H

#include <chrono>
#include <type_traits>

namespace language {
namespace threading_impl {
namespace chrono_utils {

#if __cplusplus >= 201703L
using std::chrono::ceil;
#else

namespace detail {
template <class _Tp>
struct is_duration : std::false_type {};

template <class _Rep, class _Period>
struct is_duration<std::chrono::duration<_Rep, _Period> >
  : std::true_type  {};

template <class _Rep, class _Period>
struct is_duration<const std::chrono::duration<_Rep, _Period> >
  : std::true_type  {};

template <class _Rep, class _Period>
struct is_duration<volatile std::chrono::duration<_Rep, _Period> >
  : std::true_type  {};

template <class _Rep, class _Period>
struct is_duration<const volatile std::chrono::duration<_Rep, _Period> >
  : std::true_type  {};
}

template <class To, class Rep, class Period,
          class = std::enable_if_t<detail::is_duration<To>::value>>
constexpr To
ceil(const std::chrono::duration<Rep, Period>& d)
{
  To t = std::chrono::duration_cast<To>(d);
  if (t < d)
    t = t + To{1};
  return t;
}

#endif

} // namespace chrono_utils
} // namespace threading_impl
} // namespace language

#endif // LANGUAGE_THREADING_IMPL_CHRONO_UTILS_H
