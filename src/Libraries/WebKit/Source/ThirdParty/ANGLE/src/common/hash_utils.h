/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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

//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// hash_utils.h: Hashing based helper functions.

#ifndef COMMON_HASHUTILS_H_
#define COMMON_HASHUTILS_H_

#include "common/debug.h"
#include "xxhash.h"

namespace angle
{
// Computes a hash of "key". Any data passed to this function must be multiples of
// 4 bytes, since the PMurHash32 method can only operate increments of 4-byte words.
inline size_t ComputeGenericHash(const void *key, size_t keySize)
{
    constexpr unsigned int kSeed = 0xABCDEF98;

    // We can't support "odd" alignments.  ComputeGenericHash requires aligned types
    ASSERT(keySize % 4 == 0);
#if defined(ANGLE_IS_64_BIT_CPU)
    return XXH64(key, keySize, kSeed);
#else
    return XXH32(key, keySize, kSeed);
#endif  // defined(ANGLE_IS_64_BIT_CPU)
}

template <typename T>
size_t ComputeGenericHash(const T &key)
{
    static_assert(sizeof(key) % 4 == 0, "ComputeGenericHash requires aligned types");
    return ComputeGenericHash(&key, sizeof(key));
}

inline void HashCombine(size_t &seed) {}

template <typename T, typename... Rest>
inline void HashCombine(std::size_t &seed, const T &hashableObject, Rest... rest)
{
    std::hash<T> hasher;
    seed ^= hasher(hashableObject) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    HashCombine(seed, rest...);
}

template <typename T, typename... Rest>
inline size_t HashMultiple(const T &hashableObject, Rest... rest)
{
    size_t seed = 0;
    HashCombine(seed, hashableObject, rest...);
    return seed;
}

}  // namespace angle

#endif  // COMMON_HASHUTILS_H_
