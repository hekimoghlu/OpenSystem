/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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

namespace WTF {

template<typename T>
struct RawValueTraits {
    template<typename U> using RebindTraits = RawValueTraits<U>;

    using StorageType = T;

    template<typename U>
    static ALWAYS_INLINE T exchange(StorageType& val, U&& newValue) { return std::exchange(val, newValue); }

    static ALWAYS_INLINE void swap(StorageType& a, StorageType& b) { std::swap(a, b); }
    static ALWAYS_INLINE T unwrap(const StorageType& val) { return val; }
};

} // namespace WTF

using WTF::RawValueTraits;
