/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 24, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// Test that UDT's convertible to an integral or floating point type do not
// participate in overload resolution.

#include <uscl/std/cassert>
#include <uscl/std/complex>
#include <uscl/std/type_traits>

template <class IntT>
struct UDT
{
  operator IntT() const
  {
    return 1;
  }
};

UDT<float> ft;
UDT<double> dt;
// CUDA treats long double as double
// UDT<long double> ldt;
UDT<int> it;
UDT<unsigned long> uit;

int main(int, char**)
{
  {
    cuda::std::real(ft); // expected-error {{no matching function}}
    cuda::std::real(dt); // expected-error {{no matching function}}
    // cuda::std::real(ldt); // expected-error {{no matching function}}
    cuda::std::real(it); // expected-error {{no matching function}}
    cuda::std::real(uit); // expected-error {{no matching function}}
  }
  {
    cuda::std::imag(ft); // expected-error {{no matching function}}
    cuda::std::imag(dt); // expected-error {{no matching function}}
    // cuda::std::imag(ldt); // expected-error {{no matching function}}
    cuda::std::imag(it); // expected-error {{no matching function}}
    cuda::std::imag(uit); // expected-error {{no matching function}}
  }
  {
    cuda::std::arg(ft); // expected-error {{no matching function}}
    cuda::std::arg(dt); // expected-error {{no matching function}}
    // cuda::std::arg(ldt); // expected-error {{no matching function}}
    cuda::std::arg(it); // expected-error {{no matching function}}
    cuda::std::arg(uit); // expected-error {{no matching function}}
  }
  {
    cuda::std::norm(ft); // expected-error {{no matching function}}
    cuda::std::norm(dt); // expected-error {{no matching function}}
    // cuda::std::norm(ldt); // expected-error {{no matching function}}
    cuda::std::norm(it); // expected-error {{no matching function}}
    cuda::std::norm(uit); // expected-error {{no matching function}}
  }
  {
    cuda::std::conj(ft); // expected-error {{no matching function}}
    cuda::std::conj(dt); // expected-error {{no matching function}}
    // cuda::std::conj(ldt); // expected-error {{no matching function}}
    cuda::std::conj(it); // expected-error {{no matching function}}
    cuda::std::conj(uit); // expected-error {{no matching function}}
  }
  {
    cuda::std::proj(ft); // expected-error {{no matching function}}
    cuda::std::proj(dt); // expected-error {{no matching function}}
    // cuda::std::proj(ldt); // expected-error {{no matching function}}
    cuda::std::proj(it); // expected-error {{no matching function}}
    cuda::std::proj(uit); // expected-error {{no matching function}}
  }

  return 0;
}
