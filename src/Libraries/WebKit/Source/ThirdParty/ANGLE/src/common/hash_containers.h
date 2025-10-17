/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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
// Copyright 2024 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// hash_containers.h: angle::HashMap and HashSet

#ifndef COMMON_HASH_CONTAINERS_H_
#define COMMON_HASH_CONTAINERS_H_

#if defined(ANGLE_USE_ABSEIL)
#    include "absl/container/flat_hash_map.h"
#    include "absl/container/flat_hash_set.h"
#else
#    include <unordered_map>
#    include <unordered_set>
#endif  // defined(ANGLE_USE_ABSEIL)

namespace angle
{

#if defined(ANGLE_USE_ABSEIL)
template <typename Key,
          typename T,
          class Hash = absl::container_internal::hash_default_hash<Key>,
          class Eq   = absl::container_internal::hash_default_eq<Key>>
using HashMap = absl::flat_hash_map<Key, T, Hash, Eq>;
template <typename Key,
          class Hash = absl::container_internal::hash_default_hash<Key>,
          class Eq   = absl::container_internal::hash_default_eq<Key>>
using HashSet = absl::flat_hash_set<Key, Hash, Eq>;

// Absl has generic lookup unconditionally
#    define ANGLE_HAS_HASH_MAP_GENERIC_LOOKUP 1
#else
template <typename Key,
          typename T,
          class Hash     = std::hash<Key>,
          class KeyEqual = std::equal_to<Key>>
using HashMap = std::unordered_map<Key, T, Hash, KeyEqual>;
template <typename Key, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>>
using HashSet = std::unordered_set<Key, Hash, KeyEqual>;
#    if __cpp_lib_generic_unordered_lookup >= 201811L
#        define ANGLE_HAS_HASH_MAP_GENERIC_LOOKUP 1
#    else
#        define ANGLE_HAS_HASH_MAP_GENERIC_LOOKUP 0
#    endif
#endif  // defined(ANGLE_USE_ABSEIL)

}  // namespace angle

#endif  // COMMON_HASH_CONTAINERS_H_
