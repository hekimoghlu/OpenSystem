/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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

#include <cub/detail/ptx-json/array.h>
#include <cub/detail/ptx-json/object.h>
#include <cub/detail/ptx-json/string.h>
#include <cub/detail/ptx-json/value.h>

#include <uscl/std/cstddef>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

namespace ptx_json
{
template <auto T, typename = value_traits<T>::type>
struct tagged_json;

template <int N, string<N> T, cuda::std::size_t... Is>
struct tagged_json<T, cuda::std::index_sequence<Is...>>
{
  template <typename V, typename = cuda::std::enable_if_t<is_object<V>::value || is_array<V>::value>>
  __noinline__ __device__ void operator=(V)
  {
    asm volatile("cccl.ptx_json.begin(%0)\n\n" ::"C"(storage_helper<T.str[Is]...>::value) : "memory");
    V::emit();
    asm volatile("\ncccl.ptx_json.end(%0)" ::"C"(storage_helper<T.str[Is]...>::value) : "memory");
  }
};

template <auto T>
__forceinline__ __device__ tagged_json<T> id()
{
  return {};
}
} // namespace ptx_json
