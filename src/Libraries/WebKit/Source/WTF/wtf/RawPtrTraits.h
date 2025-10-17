/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 22, 2023.
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

#include <cstdint>
#include <utility>
#include <wtf/StdLibExtras.h>

namespace WTF {
    
template<typename T>
struct RawPtrTraits {
    template<typename U> using RebindTraits = RawPtrTraits<U>;

    using StorageType = T*;

    template<typename U>
    static ALWAYS_INLINE T* exchange(StorageType& ptr, U&& newValue) { return std::exchange(ptr, newValue); }

    static ALWAYS_INLINE void swap(StorageType& a, StorageType& b) { std::swap(a, b); }
    static ALWAYS_INLINE T* unwrap(const StorageType& ptr) { return ptr; }

    static StorageType hashTableDeletedValue() { return std::bit_cast<StorageType>(static_cast<uintptr_t>(-1)); }
    static ALWAYS_INLINE bool isHashTableDeletedValue(const StorageType& ptr) { return ptr == hashTableDeletedValue(); }
};

} // namespace WTF

using WTF::RawPtrTraits;
