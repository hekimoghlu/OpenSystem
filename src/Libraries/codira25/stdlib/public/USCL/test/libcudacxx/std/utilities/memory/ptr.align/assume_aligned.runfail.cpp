/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 19, 2023.
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
#include "cuda/std/__memory/assume_aligned.h"

#include <test_macros.h>

int main(int, char**)
{
// the alignment check is disabled when it is not possible to evaluate the alignment at compile time
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  auto ptr                  = cuda::std::bit_cast<int*>(uintptr_t{0x4});
  volatile auto aligned_ptr = cuda::std::assume_aligned<64>(ptr);
  assert(ptr + 1 == aligned_ptr);
  unused(aligned_ptr);
  return 0;
#else
  return 1;
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED
}
