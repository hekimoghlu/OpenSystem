/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 18, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <uscl/std/__cccl/builtin.h>

int main(int, char**)
{
#if defined(_CCCL_BUILTIN_POPCOUNT)
  static_assert(_CCCL_BUILTIN_POPCOUNT(0b10101010) == 4);
#endif
#if defined(_CCCL_BUILTIN_POPCOUNTLL)
  static_assert(_CCCL_BUILTIN_POPCOUNTLL(0b10101010) == 4);
#endif
#if defined(_CCCL_BUILTIN_CLZ)
  static_assert(_CCCL_BUILTIN_CLZ(0b10101010) == 24);
#endif
#if defined(_CCCL_BUILTIN_CLZLL)
  static_assert(_CCCL_BUILTIN_CLZLL(0b10101010) == 56);
#endif
#if defined(_CCCL_BUILTIN_CTZ)
  static_assert(_CCCL_BUILTIN_CTZ(0b10101010) == 1);
#endif
#if defined(_CCCL_BUILTIN_CTZLL)
  static_assert(_CCCL_BUILTIN_CTZLL(0b10101010) == 1);
#endif
  return 0;
}
