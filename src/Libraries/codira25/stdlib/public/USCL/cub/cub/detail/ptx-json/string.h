/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 27, 2023.
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
#pragma once

namespace ptx_json
{
template <int N>
struct string
{
  static const constexpr auto Length = N;

  __device__ constexpr string(const char (&c)[N])
  {
    for (int i = 0; i < N; ++i)
    {
      str[i] = c[i];
    }
    (void) Length;
  }

  char str[N];
};

__forceinline__ __device__ void comma()
{
  asm volatile("," ::: "memory");
}

#pragma nv_diag_suppress 177
template <char... Cs>
struct storage_helper
{
  // This, and the dance to invoke this through value_traits elsewhere, is necessary because the "C" inline assembly
  // constraint supported by NVCC requires that its argument is a pointer to a constant array of type char; NVCC also
  // doesn't allow passing raw character literals as pointer template arguments; and *also* it seems to look at the type
  // of a containing object, not a subobject it is given, when passed in a pointer to an array inside a literal type.
  // All of this means that we can't just pass strings, and *also* we can't just use the string<N>::array member above
  // as the string literal; therefore, using the fact that the length of the string is a core constant expression in the
  // definition of value_traits, we can generate a variadic pack that allows us to expand the contents of
  // string<N>::array into a comma separated list of N chars. We can then plug that in as template arguments to
  // storage_helper, which then can, as below, turn that into its own char array that NVCC accepts as an argument for a
  // "C" inline assembly constraint.
  static const constexpr char value[] = {Cs...};
};
#pragma nv_diag_default 177
} // namespace ptx_json
