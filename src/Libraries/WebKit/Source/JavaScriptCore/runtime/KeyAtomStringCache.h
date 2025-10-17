/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 16, 2025.
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

#include <wtf/text/AtomStringImpl.h>

namespace JSC {

class VM;

class KeyAtomStringCache {
public:
    static constexpr auto maxStringLengthForCache = 64;
    static constexpr auto capacity = 512;
    using Cache = std::array<JSString*, capacity>;

    template<typename Buffer, typename Func>
    JSString* make(VM&, Buffer&, const Func&);

    ALWAYS_INLINE void clear()
    {
        m_cache.fill({ });
    }

private:
    Cache m_cache { };
};

} // namespace JSC
