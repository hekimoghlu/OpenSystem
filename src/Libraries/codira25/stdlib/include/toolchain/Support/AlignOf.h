/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 22, 2024.
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

//===--- AlignOf.h - Portable calculation of type alignment -----*- C++ -*-===//
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
// This file defines the AlignedCharArrayUnion class.
//
//===----------------------------------------------------------------------===//

#ifndef TOOLCHAIN_SUPPORT_ALIGNOF_H
#define TOOLCHAIN_SUPPORT_ALIGNOF_H

#include <type_traits>

inline namespace __language { inline namespace __runtime {
namespace toolchain {

/// A suitably aligned and sized character array member which can hold elements
/// of any type.
///
/// This template is equivalent to std::aligned_union_t<1, ...>, but we cannot
/// use it due to a bug in the MSVC x86 compiler:
/// https://github.com/microsoft/STL/issues/1533
/// Using `alignas` here works around the bug.
template <typename T, typename... Ts> struct AlignedCharArrayUnion {
  using AlignedUnion = std::aligned_union_t<1, T, Ts...>;
  alignas(alignof(AlignedUnion)) char buffer[sizeof(AlignedUnion)];
};

} // end namespace toolchain
}} // namespace language::runtime

#endif // TOOLCHAIN_SUPPORT_ALIGNOF_H
