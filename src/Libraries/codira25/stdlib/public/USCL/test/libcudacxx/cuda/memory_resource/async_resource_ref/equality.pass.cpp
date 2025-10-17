/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc

// cuda::mr::resource_ref equality

#include <uscl/memory_resource>
#include <uscl/std/cassert>
#include <uscl/std/cstdint>
#include <uscl/stream_ref>

#include "types.h"

using ref = cuda::mr::resource_ref<cuda::mr::host_accessible,
                                   property_with_value<int>,
                                   property_with_value<double>,
                                   property_without_value<std::size_t>>;

using pertubed_properties =
  cuda::mr::resource_ref<cuda::mr::host_accessible,
                         property_with_value<double>,
                         property_with_value<int>,
                         property_without_value<std::size_t>>;

using res = test_resource<cuda::mr::host_accessible,
                          property_with_value<int>,
                          property_with_value<double>,
                          property_without_value<std::size_t>>;
using other_res =
  test_resource<cuda::mr::host_accessible,
                property_with_value<double>,
                property_with_value<int>,
                property_without_value<std::size_t>>;

void test_equality()
{
  res input{42};
  res with_equal_value{42};
  res with_different_value{1337};

  assert(input == with_equal_value);
  assert(input != with_different_value);

  assert(ref{input} == ref{with_equal_value});
  assert(ref{input} != ref{with_different_value});

  // Should ignore pertubed properties
  assert(ref{input} == pertubed_properties{with_equal_value});
  assert(ref{input} != pertubed_properties{with_different_value});

  // Should reject different resources
  other_res other_with_matching_value{42};
  other_res other_with_different_value{1337};
  assert(ref{input} != ref{other_with_matching_value});
  assert(ref{input} != ref{other_with_different_value});

  assert(ref{input} != pertubed_properties{other_with_matching_value});
  assert(ref{input} != pertubed_properties{other_with_matching_value});
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_equality();))

  return 0;
}
