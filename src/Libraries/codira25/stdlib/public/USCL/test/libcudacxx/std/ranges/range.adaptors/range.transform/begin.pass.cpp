/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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
// constexpr iterator<false> begin();
// constexpr iterator<true> begin() const
//   requires range<const V> &&
//            regular_invocable<const F&, range_reference_t<const V>>;

#include <uscl/std/ranges>

#include "test_macros.h"
#include "types.h"

template <class T>
_CCCL_CONCEPT BeginInvocable = _CCCL_REQUIRES_EXPR((T), T t)((t.begin()));

__host__ __device__ constexpr bool test()
{
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  {
    cuda::std::ranges::transform_view transformView(MoveOnlyView{buff}, PlusOneMutable{});
    assert(transformView.begin().base() == buff);
    assert(*transformView.begin() == 1);
  }

  {
    cuda::std::ranges::transform_view transformView(ForwardView{buff}, PlusOneMutable{});
    assert(base(transformView.begin().base()) == buff);
    assert(*transformView.begin() == 1);
  }

  {
    cuda::std::ranges::transform_view transformView(InputView{buff}, PlusOneMutable{});
    assert(base(transformView.begin().base()) == buff);
    assert(*transformView.begin() == 1);
  }

  {
    const cuda::std::ranges::transform_view transformView(MoveOnlyView{buff}, PlusOne{});
    assert(*transformView.begin() == 1);
  }

  static_assert(!BeginInvocable<const cuda::std::ranges::transform_view<MoveOnlyView, PlusOneMutable>>);

  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_ADDRESSOF

  return 0;
}
