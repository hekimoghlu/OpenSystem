/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 16, 2021.
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

#include <uscl/std/limits>

#include <memory>
#include <ostream>

namespace c2h
{

struct custom_type_state_t
{
  std::size_t key{};
  std::size_t val{};
};

template <template <typename> class... Policies>
class custom_type_t
    : public custom_type_state_t
    , public Policies<custom_type_t<Policies...>>...
{
public:
  friend __host__ std::ostream& operator<<(std::ostream& os, const custom_type_t& self)
  {
    return os << "{ " << self.key << ", " << self.val << " }";
  }
};

template <std::size_t TotalSize>
struct huge_data
{
  template <class CustomType>
  class type
  {
    static constexpr auto extra_member_bytes = (TotalSize - sizeof(custom_type_state_t));
    std::uint8_t data[extra_member_bytes];
  };
};

template <class CustomType>
class less_comparable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the
  // Microsoft host compiler does not.
  char workaround_msvc{};

public:
  friend __host__ __device__ bool operator<(const CustomType& lhs, const CustomType& rhs)
  {
    return lhs.key < rhs.key;
  }
};

template <class CustomType>
class greater_comparable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the
  // Microsoft host compiler does not.
  char workaround_msvc{};

public:
  friend __host__ __device__ bool operator>(const CustomType& lhs, const CustomType& rhs)
  {
    return lhs.key > rhs.key;
  }
};

template <class CustomType>
class lexicographical_less_comparable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the
  // Microsoft host compiler does not.
  char workaround_msvc{};

public:
  friend __host__ __device__ bool operator<(const CustomType& lhs, const CustomType& rhs)
  {
    return lhs.key == rhs.key ? lhs.val < rhs.val : lhs.key < rhs.key;
  }
};

template <class CustomType>
class lexicographical_greater_comparable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the
  // Microsoft host compiler does not.
  char workaround_msvc{};

public:
  friend __host__ __device__ bool operator>(const CustomType& lhs, const CustomType& rhs)
  {
    return lhs.key == rhs.key ? lhs.val > rhs.val : lhs.key > rhs.key;
  }
};

template <class CustomType>
class equal_comparable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the
  // Microsoft host compiler does not.
  char workaround_msvc{};

public:
  friend __host__ __device__ bool operator==(const CustomType& lhs, const CustomType& rhs)
  {
    return lhs.key == rhs.key && lhs.val == rhs.val;
  }

  friend __host__ __device__ bool operator!=(const CustomType& lhs, const CustomType& rhs)
  {
    return !(lhs == rhs);
  }
};

template <class CustomType>
class subtractable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the
  // Microsoft host compiler does not.
  char workaround_msvc{};

public:
  friend __host__ __device__ CustomType operator-(const CustomType& lhs, const CustomType& rhs)
  {
    CustomType result{};

    result.key = lhs.key - rhs.key;
    result.val = lhs.val - rhs.val;

    return result;
  }
};

template <class CustomType>
class accumulateable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the
  // Microsoft host compiler does not.
  char workaround_msvc{};

public:
  friend __host__ __device__ CustomType operator+(const CustomType& lhs, const CustomType& rhs)
  {
    CustomType result{};

    result.key = lhs.key + rhs.key;
    result.val = lhs.val + rhs.val;

    return result;
  }
};

} // namespace c2h

_CCCL_BEGIN_NAMESPACE_CUDA_STD
template <template <typename> class... Policies>
class numeric_limits<c2h::custom_type_t<Policies...>>
{
public:
  static constexpr bool is_specialized = true;

  static __host__ __device__ c2h::custom_type_t<Policies...> max()
  {
    c2h::custom_type_t<Policies...> val;
    val.key = numeric_limits<std::size_t>::max();
    val.val = numeric_limits<std::size_t>::max();
    return val;
  }

  static __host__ __device__ c2h::custom_type_t<Policies...> min()
  {
    c2h::custom_type_t<Policies...> val;
    val.key = numeric_limits<std::size_t>::min();
    val.val = numeric_limits<std::size_t>::min();
    return val;
  }

  static __host__ __device__ c2h::custom_type_t<Policies...> lowest()
  {
    c2h::custom_type_t<Policies...> val;
    val.key = numeric_limits<std::size_t>::lowest();
    val.val = numeric_limits<std::size_t>::lowest();
    return val;
  }
};
_CCCL_END_NAMESPACE_CUDA_STD
