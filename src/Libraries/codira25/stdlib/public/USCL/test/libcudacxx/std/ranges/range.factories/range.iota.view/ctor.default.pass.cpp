/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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
// iota_view() requires default_initializable<W> = default;

#include <uscl/std/cassert>
#include <uscl/std/ranges>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::iota_view<Int42<DefaultTo42>> io{};
    assert((*io.begin()).value_ == 42);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  static_assert(!cuda::std::default_initializable<Int42<ValueCtor>>);
  static_assert(cuda::std::default_initializable<Int42<DefaultTo42>>);

  return 0;
}
