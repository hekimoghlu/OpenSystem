/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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

#ifndef POINTER_COMPARISON_TEST_HELPER_HPP
#define POINTER_COMPARISON_TEST_HELPER_HPP

#include <uscl/std/cassert>

#include <cstdint>
#include <memory>
#include <vector>

#include "test_macros.h"

template <class T, template <class> class CompareTemplate>
void do_pointer_comparison_test()
{
  typedef CompareTemplate<T*> Compare;
  typedef CompareTemplate<std::uintptr_t> UIntCompare;
  typedef CompareTemplate<void> VoidCompare;

  std::vector<std::shared_ptr<T>> pointers;
  const std::size_t test_size = 100;
  for (size_t i = 0; i < test_size; ++i)
  {
    pointers.push_back(std::shared_ptr<T>(new T()));
  }
  Compare comp;
  UIntCompare ucomp;
  VoidCompare vcomp;
  for (size_t i = 0; i < test_size; ++i)
  {
    for (size_t j = 0; j < test_size; ++j)
    {
      T* lhs                  = pointers[i].get();
      T* rhs                  = pointers[j].get();
      std::uintptr_t lhs_uint = reinterpret_cast<std::uintptr_t>(lhs);
      std::uintptr_t rhs_uint = reinterpret_cast<std::uintptr_t>(rhs);
      assert(comp(lhs, rhs) == ucomp(lhs_uint, rhs_uint));
      assert(vcomp(lhs, rhs) == ucomp(lhs_uint, rhs_uint));
    }
  }
}

#endif // POINTER_COMPARISON_TEST_HELPER_HPP
