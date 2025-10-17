/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 21, 2021.
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

//===--- HeaderFooterLayout.h -----------------------------------*- C++ -*-===//
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

#ifndef LANGUAGE_BASIC_HEADER_FOOTER_LAYOUT_H
#define LANGUAGE_BASIC_HEADER_FOOTER_LAYOUT_H

#include <cstddef>

namespace language {

template <class T>
class size_without_trailing_padding {
  struct ExtraByte { char _size_without_trailing_padding_probe; };
  struct Probe: T, ExtraByte {};
public:
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winvalid-offsetof"
  enum { value = offsetof(Probe, _size_without_trailing_padding_probe) };
#pragma clang diagnostic pop
};

namespace detail {

template <ptrdiff_t size>
struct LayoutPadding {
  char padding[size];
};
template <>
struct LayoutPadding<0> {};

template <class Header, class Footer, size_t TotalSize>
struct HeaderFooterLayoutPaddingSize {
  enum : ptrdiff_t {
    maxFooterOffset = TotalSize - (ptrdiff_t)size_without_trailing_padding<Footer>::value,
    footerAlignment = (ptrdiff_t)alignof(Footer),
    footerOffset = maxFooterOffset - (maxFooterOffset % footerAlignment),
    value = footerOffset - (ptrdiff_t)size_without_trailing_padding<Header>::value
  };
};

} // namespace detail

template <class Header, class Footer, size_t TotalSize>
struct HeaderFooterLayout
    : Header,
      detail::LayoutPadding<detail::HeaderFooterLayoutPaddingSize<
          Header, Footer, TotalSize>::value>,
      Footer {};

} // namespace language

#endif // LANGUAGE_BASIC_HEADER_FOOTER_LAYOUT_H

