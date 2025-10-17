/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 17, 2025.
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

#include "Identifier.h"
#include "KeyAtomStringCache.h"
#include "SmallStrings.h"
#include "VM.h"

namespace JSC {

template<typename Buffer, typename Func>
ALWAYS_INLINE JSString* KeyAtomStringCache::make(VM& vm, Buffer& buffer, const Func& func)
{
    if (buffer.characters.empty())
        return jsEmptyString(vm);

    if (buffer.characters.size() == 1) {
        auto firstCharacter = buffer.characters[0];
        if (firstCharacter <= maxSingleCharacterString)
            return vm.smallStrings.singleCharacterString(firstCharacter);
    }

    ASSERT(buffer.characters.size() <= maxStringLengthForCache);
    auto& slot = m_cache[buffer.hash % capacity];
    if (slot) {
        auto* impl = slot->tryGetValueImpl();
        if (impl->hash() == buffer.hash && equal(impl, buffer.characters))
            return slot;
    }

    JSString* result = func(vm, buffer);
    if (LIKELY(result))
        slot = result;
    return result;
}

} // namespace JSC
