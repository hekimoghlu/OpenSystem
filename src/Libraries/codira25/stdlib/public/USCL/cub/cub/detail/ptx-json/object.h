/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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
#include <cub/detail/ptx-json/value.h>

#include <uscl/std/type_traits>

namespace ptx_json
{
template <auto K, typename V>
struct keyed_value
{
  __forceinline__ __device__ static void emit()
  {
    value<K>::emit();
    asm volatile(":" ::: "memory");
    V::emit();
  }
};

template <typename T>
struct is_keyed_value : cuda::std::false_type
{};

template <auto K, typename V>
struct is_keyed_value<keyed_value<K, V>> : cuda::std::true_type
{};

template <typename T>
concept a_keyed_value = is_keyed_value<T>::value;

template <auto... KV>
struct object;

template <>
struct object<>
{
  __forceinline__ __device__ static void emit()
  {
    asm volatile("{}" ::: "memory");
  }
};

template <a_keyed_value auto First, a_keyed_value auto... KVs>
struct object<First, KVs...>
{
  __forceinline__ __device__ static void emit()
  {
    asm volatile("{" ::: "memory");
    First.emit();
    ((comma(), KVs.emit()), ...);
    asm volatile("}" ::: "memory");
  }
};

template <typename T>
struct is_object : cuda::std::false_type
{};

template <auto... KV>
struct is_object<object<KV...>> : cuda::std::true_type
{};

template <string V>
struct key
{
  template <typename U>
  __forceinline__ __device__ constexpr keyed_value<V, U> operator=(U)
  {
    return {};
  }
};
} // namespace ptx_json
