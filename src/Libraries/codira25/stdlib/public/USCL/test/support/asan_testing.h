/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 15, 2023.
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

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASAN_TESTING_H
#define ASAN_TESTING_H

#include "test_macros.h"

#if _CCCL_HAS_FEATURE(address_sanitizer)
extern "C" int __sanitizer_verify_contiguous_container(const void* beg, const void* mid, const void* end);

template <typename T, typename Alloc>
bool is_contiguous_container_asan_correct(const std::vector<T, Alloc>& c)
{
  if (std::is_same<Alloc, std::allocator<T>>::value && c.data() != nullptr)
  {
    return __sanitizer_verify_contiguous_container(c.data(), c.data() + c.size(), c.data() + c.capacity()) != 0;
  }
  return true;
}

#else
template <typename T, typename Alloc>
bool is_contiguous_container_asan_correct(const std::vector<T, Alloc>&)
{
  return true;
}
#endif

#endif // ASAN_TESTING_H
