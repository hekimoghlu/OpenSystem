/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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

#include <cub/detail/ptx-json/string.h>

#include <uscl/std/type_traits>

namespace ptx_json
{
template <auto V>
struct value_traits
{
  using type = void;
};

template <auto V, typename = value_traits<V>::type>
struct value;

#pragma nv_diag_suppress 177
template <int N, string<N> V>
struct value_traits<V>
{
  using type = cuda::std::make_index_sequence<N>;
};
#pragma nv_diag_default 177

template <typename T>
struct is_value : cuda::std::false_type
{};

template <auto V>
struct is_value<value<V>> : cuda::std::true_type
{};

template <typename T>
concept a_value = is_value<T>::value;

template <a_value auto Nested>
struct value<Nested, void>
{
  __forceinline__ __device__ static void emit()
  {
    value<Nested>::emit();
  }
};

template <int V>
struct value<V, void>
{
  __forceinline__ __device__ static void emit()
  {
    asm volatile("%0" ::"n"(V) : "memory");
  }
};

#pragma nv_diag_suppress 842
template <int N, string<N> V, cuda::std::size_t... Is>
struct value<V, cuda::std::index_sequence<Is...>>
{
#pragma nv_diag_default 842
  __forceinline__ __device__ static void emit()
  {
    // See the definition of storage_helper for why laundering the string through it is necessary.
    asm volatile("\"%0\"" ::"C"(storage_helper<V.str[Is]...>::value) : "memory");
  }
};
}; // namespace ptx_json
