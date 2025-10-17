/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 5, 2023.
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

#include <wtf/HashMap.h>
#include <wtf/RobinHoodHashTable.h>

namespace WTF {

// 95% load-factor.
template<typename KeyArg, typename MappedArg, typename HashArg = DefaultHash<KeyArg>, typename KeyTraitsArg = HashTraits<KeyArg>, typename MappedTraitsArg = HashTraits<MappedArg>>
using MemoryCompactLookupOnlyRobinHoodHashMap = UncheckedKeyHashMap<KeyArg, MappedArg, HashArg, KeyTraitsArg, MappedTraitsArg, MemoryCompactLookupOnlyRobinHoodHashTableTraits>;

// 90% load-factor.
template<typename KeyArg, typename MappedArg, typename HashArg = DefaultHash<KeyArg>, typename KeyTraitsArg = HashTraits<KeyArg>, typename MappedTraitsArg = HashTraits<MappedArg>>
using MemoryCompactRobinHoodHashMap = UncheckedKeyHashMap<KeyArg, MappedArg, HashArg, KeyTraitsArg, MappedTraitsArg, MemoryCompactRobinHoodHashTableTraits>;

// 75% load-factor.
template<typename KeyArg, typename MappedArg, typename HashArg = DefaultHash<KeyArg>, typename KeyTraitsArg = HashTraits<KeyArg>, typename MappedTraitsArg = HashTraits<MappedArg>>
using FastRobinHoodHashMap = UncheckedKeyHashMap<KeyArg, MappedArg, HashArg, KeyTraitsArg, MappedTraitsArg, FastRobinHoodHashTableTraits>;

} // namespace WTF

using WTF::MemoryCompactLookupOnlyRobinHoodHashMap;
using WTF::MemoryCompactRobinHoodHashMap;
using WTF::FastRobinHoodHashMap;
