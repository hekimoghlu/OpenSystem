/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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

//===--- hlsl_compat_overloads.h - Extra HLSL overloads for intrinsics ----===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
// 
// Author: Tunjay Akbarli
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_COMPAT_OVERLOADS_H_
#define _HLSl_COMPAT_OVERLOADS_H_

namespace hlsl {

// Note: Functions in this file are sorted alphabetically, then grouped by base
// element type, and the element types are sorted by size, then singed integer,
// unsigned integer and floating point. Keeping this ordering consistent will
// help keep this file manageable as it grows.

#define _DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(fn)                                 \
  constexpr float fn(double V) { return fn((float)V); }                        \
  constexpr float2 fn(double2 V) { return fn((float2)V); }                     \
  constexpr float3 fn(double3 V) { return fn((float3)V); }                     \
  constexpr float4 fn(double4 V) { return fn((float4)V); }

#define _DXC_COMPAT_BINARY_DOUBLE_OVERLOADS(fn)                                \
  constexpr float fn(double V1, double V2) {                                   \
    return fn((float)V1, (float)V2);                                           \
  }                                                                            \
  constexpr float2 fn(double2 V1, double2 V2) {                                \
    return fn((float2)V1, (float2)V2);                                         \
  }                                                                            \
  constexpr float3 fn(double3 V1, double3 V2) {                                \
    return fn((float3)V1, (float3)V2);                                         \
  }                                                                            \
  constexpr float4 fn(double4 V1, double4 V2) {                                \
    return fn((float4)V1, (float4)V2);                                         \
  }

#define _DXC_COMPAT_TERNARY_DOUBLE_OVERLOADS(fn)                               \
  constexpr float fn(double V1, double V2, double V3) {                        \
    return fn((float)V1, (float)V2, (float)V3);                                \
  }                                                                            \
  constexpr float2 fn(double2 V1, double2 V2, double2 V3) {                    \
    return fn((float2)V1, (float2)V2, (float2)V3);                             \
  }                                                                            \
  constexpr float3 fn(double3 V1, double3 V2, double3 V3) {                    \
    return fn((float3)V1, (float3)V2, (float3)V3);                             \
  }                                                                            \
  constexpr float4 fn(double4 V1, double4 V2, double4 V3) {                    \
    return fn((float4)V1, (float4)V2, (float4)V3);                             \
  }

#define _DXC_COMPAT_UNARY_INTEGER_OVERLOADS(fn)                                \
  constexpr float fn(int V) { return fn((float)V); }                           \
  constexpr float2 fn(int2 V) { return fn((float2)V); }                        \
  constexpr float3 fn(int3 V) { return fn((float3)V); }                        \
  constexpr float4 fn(int4 V) { return fn((float4)V); }                        \
  constexpr float fn(uint V) { return fn((float)V); }                          \
  constexpr float2 fn(uint2 V) { return fn((float2)V); }                       \
  constexpr float3 fn(uint3 V) { return fn((float3)V); }                       \
  constexpr float4 fn(uint4 V) { return fn((float4)V); }                       \
  constexpr float fn(int64_t V) { return fn((float)V); }                       \
  constexpr float2 fn(int64_t2 V) { return fn((float2)V); }                    \
  constexpr float3 fn(int64_t3 V) { return fn((float3)V); }                    \
  constexpr float4 fn(int64_t4 V) { return fn((float4)V); }                    \
  constexpr float fn(uint64_t V) { return fn((float)V); }                      \
  constexpr float2 fn(uint64_t2 V) { return fn((float2)V); }                   \
  constexpr float3 fn(uint64_t3 V) { return fn((float3)V); }                   \
  constexpr float4 fn(uint64_t4 V) { return fn((float4)V); }

#define _DXC_COMPAT_BINARY_INTEGER_OVERLOADS(fn)                               \
  constexpr float fn(int V1, int V2) { return fn((float)V1, (float)V2); }      \
  constexpr float2 fn(int2 V1, int2 V2) { return fn((float2)V1, (float2)V2); } \
  constexpr float3 fn(int3 V1, int3 V2) { return fn((float3)V1, (float3)V2); } \
  constexpr float4 fn(int4 V1, int4 V2) { return fn((float4)V1, (float4)V2); } \
  constexpr float fn(uint V1, uint V2) { return fn((float)V1, (float)V2); }    \
  constexpr float2 fn(uint2 V1, uint2 V2) {                                    \
    return fn((float2)V1, (float2)V2);                                         \
  }                                                                            \
  constexpr float3 fn(uint3 V1, uint3 V2) {                                    \
    return fn((float3)V1, (float3)V2);                                         \
  }                                                                            \
  constexpr float4 fn(uint4 V1, uint4 V2) {                                    \
    return fn((float4)V1, (float4)V2);                                         \
  }                                                                            \
  constexpr float fn(int64_t V1, int64_t V2) {                                 \
    return fn((float)V1, (float)V2);                                           \
  }                                                                            \
  constexpr float2 fn(int64_t2 V1, int64_t2 V2) {                              \
    return fn((float2)V1, (float2)V2);                                         \
  }                                                                            \
  constexpr float3 fn(int64_t3 V1, int64_t3 V2) {                              \
    return fn((float3)V1, (float3)V2);                                         \
  }                                                                            \
  constexpr float4 fn(int64_t4 V1, int64_t4 V2) {                              \
    return fn((float4)V1, (float4)V2);                                         \
  }                                                                            \
  constexpr float fn(uint64_t V1, uint64_t V2) {                               \
    return fn((float)V1, (float)V2);                                           \
  }                                                                            \
  constexpr float2 fn(uint64_t2 V1, uint64_t2 V2) {                            \
    return fn((float2)V1, (float2)V2);                                         \
  }                                                                            \
  constexpr float3 fn(uint64_t3 V1, uint64_t3 V2) {                            \
    return fn((float3)V1, (float3)V2);                                         \
  }                                                                            \
  constexpr float4 fn(uint64_t4 V1, uint64_t4 V2) {                            \
    return fn((float4)V1, (float4)V2);                                         \
  }

#define _DXC_COMPAT_TERNARY_INTEGER_OVERLOADS(fn)                              \
  constexpr float fn(int V1, int V2, int V3) {                                 \
    return fn((float)V1, (float)V2, (float)V3);                                \
  }                                                                            \
  constexpr float2 fn(int2 V1, int2 V2, int2 V3) {                             \
    return fn((float2)V1, (float2)V2, (float2)V3);                             \
  }                                                                            \
  constexpr float3 fn(int3 V1, int3 V2, int3 V3) {                             \
    return fn((float3)V1, (float3)V2, (float3)V3);                             \
  }                                                                            \
  constexpr float4 fn(int4 V1, int4 V2, int4 V3) {                             \
    return fn((float4)V1, (float4)V2, (float4)V3);                             \
  }                                                                            \
  constexpr float fn(uint V1, uint V2, uint V3) {                              \
    return fn((float)V1, (float)V2, (float)V3);                                \
  }                                                                            \
  constexpr float2 fn(uint2 V1, uint2 V2, uint2 V3) {                          \
    return fn((float2)V1, (float2)V2, (float2)V3);                             \
  }                                                                            \
  constexpr float3 fn(uint3 V1, uint3 V2, uint3 V3) {                          \
    return fn((float3)V1, (float3)V2, (float3)V3);                             \
  }                                                                            \
  constexpr float4 fn(uint4 V1, uint4 V2, uint4 V3) {                          \
    return fn((float4)V1, (float4)V2, (float4)V3);                             \
  }                                                                            \
  constexpr float fn(int64_t V1, int64_t V2, int64_t V3) {                     \
    return fn((float)V1, (float)V2, (float)V3);                                \
  }                                                                            \
  constexpr float2 fn(int64_t2 V1, int64_t2 V2, int64_t2 V3) {                 \
    return fn((float2)V1, (float2)V2, (float2)V3);                             \
  }                                                                            \
  constexpr float3 fn(int64_t3 V1, int64_t3 V2, int64_t3 V3) {                 \
    return fn((float3)V1, (float3)V2, (float3)V3);                             \
  }                                                                            \
  constexpr float4 fn(int64_t4 V1, int64_t4 V2, int64_t4 V3) {                 \
    return fn((float4)V1, (float4)V2, (float4)V3);                             \
  }                                                                            \
  constexpr float fn(uint64_t V1, uint64_t V2, uint64_t V3) {                  \
    return fn((float)V1, (float)V2, (float)V3);                                \
  }                                                                            \
  constexpr float2 fn(uint64_t2 V1, uint64_t2 V2, uint64_t2 V3) {              \
    return fn((float2)V1, (float2)V2, (float2)V3);                             \
  }                                                                            \
  constexpr float3 fn(uint64_t3 V1, uint64_t3 V2, uint64_t3 V3) {              \
    return fn((float3)V1, (float3)V2, (float3)V3);                             \
  }                                                                            \
  constexpr float4 fn(uint64_t4 V1, uint64_t4 V2, uint64_t4 V3) {              \
    return fn((float4)V1, (float4)V2, (float4)V3);                             \
  }

//===----------------------------------------------------------------------===//
// acos builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(acos)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(acos)

//===----------------------------------------------------------------------===//
// asin builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(asin)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(asin)

//===----------------------------------------------------------------------===//
// atan builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(atan)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(atan)

//===----------------------------------------------------------------------===//
// atan2 builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_BINARY_DOUBLE_OVERLOADS(atan2)
_DXC_COMPAT_BINARY_INTEGER_OVERLOADS(atan2)

//===----------------------------------------------------------------------===//
// ceil builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(ceil)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(ceil)

//===----------------------------------------------------------------------===//
// clamp builtins overloads
//===----------------------------------------------------------------------===//

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
clamp(vector<T, N> p0, vector<T, N> p1, T p2) {
  return clamp(p0, p1, (vector<T, N>)p2);
}

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
clamp(vector<T, N> p0, T p1, vector<T, N> p2) {
  return clamp(p0, (vector<T, N>)p1, p2);
}

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
clamp(vector<T, N> p0, T p1, T p2) {
  return clamp(p0, (vector<T, N>)p1, (vector<T, N>)p2);
}

//===----------------------------------------------------------------------===//
// cos builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(cos)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(cos)

//===----------------------------------------------------------------------===//
// cosh builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(cosh)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(cosh)

//===----------------------------------------------------------------------===//
// degrees builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(degrees)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(degrees)

//===----------------------------------------------------------------------===//
// exp builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(exp)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(exp)

//===----------------------------------------------------------------------===//
// exp2 builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(exp2)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(exp2)

//===----------------------------------------------------------------------===//
// floor builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(floor)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(floor)

//===----------------------------------------------------------------------===//
// frac builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(frac)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(frac)

//===----------------------------------------------------------------------===//
// isinf builtins overloads
//===----------------------------------------------------------------------===//

constexpr bool isinf(double V) { return isinf((float)V); }
constexpr bool2 isinf(double2 V) { return isinf((float2)V); }
constexpr bool3 isinf(double3 V) { return isinf((float3)V); }
constexpr bool4 isinf(double4 V) { return isinf((float4)V); }

//===----------------------------------------------------------------------===//
// lerp builtins overloads
//===----------------------------------------------------------------------===//

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
lerp(vector<T, N> x, vector<T, N> y, T s) {
  return lerp(x, y, (vector<T, N>)s);
}

_DXC_COMPAT_TERNARY_DOUBLE_OVERLOADS(lerp)
_DXC_COMPAT_TERNARY_INTEGER_OVERLOADS(lerp)

//===----------------------------------------------------------------------===//
// log builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(log)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(log)

//===----------------------------------------------------------------------===//
// log10 builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(log10)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(log10)

//===----------------------------------------------------------------------===//
// log2 builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(log2)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(log2)

//===----------------------------------------------------------------------===//
// max builtins overloads
//===----------------------------------------------------------------------===//

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
max(vector<T, N> p0, T p1) {
  return max(p0, (vector<T, N>)p1);
}

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
max(T p0, vector<T, N> p1) {
  return max((vector<T, N>)p0, p1);
}

//===----------------------------------------------------------------------===//
// min builtins overloads
//===----------------------------------------------------------------------===//

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
min(vector<T, N> p0, T p1) {
  return min(p0, (vector<T, N>)p1);
}

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
min(T p0, vector<T, N> p1) {
  return min((vector<T, N>)p0, p1);
}

//===----------------------------------------------------------------------===//
// normalize builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(normalize)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(normalize)

//===----------------------------------------------------------------------===//
// pow builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_BINARY_DOUBLE_OVERLOADS(pow)
_DXC_COMPAT_BINARY_INTEGER_OVERLOADS(pow)

//===----------------------------------------------------------------------===//
// rsqrt builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(rsqrt)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(rsqrt)

//===----------------------------------------------------------------------===//
// round builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(round)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(round)

//===----------------------------------------------------------------------===//
// sin builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(sin)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(sin)

//===----------------------------------------------------------------------===//
// sinh builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(sinh)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(sinh)

//===----------------------------------------------------------------------===//
// sqrt builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(sqrt)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(sqrt)

//===----------------------------------------------------------------------===//
// step builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_BINARY_DOUBLE_OVERLOADS(step)
_DXC_COMPAT_BINARY_INTEGER_OVERLOADS(step)

//===----------------------------------------------------------------------===//
// tan builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(tan)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(tan)

//===----------------------------------------------------------------------===//
// tanh builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(tanh)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(tanh)

//===----------------------------------------------------------------------===//
// trunc builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(trunc)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(trunc)

//===----------------------------------------------------------------------===//
// radians builtins overloads
//===----------------------------------------------------------------------===//

_DXC_COMPAT_UNARY_DOUBLE_OVERLOADS(radians)
_DXC_COMPAT_UNARY_INTEGER_OVERLOADS(radians)

} // namespace hlsl
#endif // _HLSL_COMPAT_OVERLOADS_H_
