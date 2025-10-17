/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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

#include <uscl/std/cassert>
#include <uscl/std/complex>
#include <uscl/std/linalg>
#include <uscl/std/mdspan>
#include <uscl/std/type_traits>

int main(int, char**)
{
  using dynamic_extents = cuda::std::dextents<size_t, 2>;
  using complex_t       = cuda::std::complex<float>;
  {
    cuda::std::array<complex_t, 6> d{
      complex_t{2.f, 3.f},
      complex_t{4.f, 5.f},
      complex_t{6.f, 7.f},
      complex_t{8.f, 9.f},
      complex_t{10.f, 11.f},
      complex_t{12.f, 13.f},
    };
    //     42, 43, 44
    //     45, 46, 47
    cuda::std::mdspan<complex_t, dynamic_extents> md(d.data(), 2, 3);
    auto conj_transposed_md = cuda::std::linalg::conjugate_transposed(md);

    assert(conj_transposed_md.static_extent(0) == cuda::std::dynamic_extent);
    assert(conj_transposed_md.static_extent(1) == cuda::std::dynamic_extent);
    assert(conj_transposed_md.extent(0) == 3);
    assert(conj_transposed_md.extent(1) == 2);
    assert(cuda::std::conj(md(0, 0)) == conj_transposed_md(0, 0));
    assert(cuda::std::conj(md(0, 1)) == conj_transposed_md(1, 0));
    assert(cuda::std::conj(md(0, 2)) == conj_transposed_md(2, 0));
    assert(cuda::std::conj(md(1, 0)) == conj_transposed_md(0, 1));
    assert(cuda::std::conj(md(1, 1)) == conj_transposed_md(1, 1));
    assert(cuda::std::conj(md(1, 2)) == conj_transposed_md(2, 1));
  }
  return 0;
}
