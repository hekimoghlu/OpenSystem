/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <uscl/bit>
#include <uscl/std/cassert>
#include <uscl/std/cstdint>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using T = uint32_t;
  static_assert(cuda::bitfield_insert(T{0}, T{0}, -1, 1));
  static_assert(cuda::bitfield_insert(T{0}, T{0}, 0, -1));
  static_assert(cuda::bitfield_insert(T{0}, T{0}, 0, 33));
  static_assert(cuda::bitfield_insert(T{0}, T{0}, 32, 1));
  static_assert(cuda::bitfield_insert(T{0}, T{0}, 20, 20));

  static_assert(cuda::bitfield_extract(T{0}, -1, 1));
  static_assert(cuda::bitfield_extract(T{0}, 0, -1));
  static_assert(cuda::bitfield_extract(T{0}, 0, 33));
  static_assert(cuda::bitfield_extract(T{0}, 32, 1));
  static_assert(cuda::bitfield_extract(T{0}, 20, 20));
  return 0;
}
