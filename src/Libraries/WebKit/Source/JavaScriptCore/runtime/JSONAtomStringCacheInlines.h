/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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
#include "JSONAtomStringCache.h"
#include "SmallStrings.h"
#include "VM.h"

namespace JSC {

// FIXME: This should take in a std::span.
template<typename CharacterType>
ALWAYS_INLINE Ref<AtomStringImpl> JSONAtomStringCache::make(std::span<const CharacterType> characters)
{
    if (characters.empty())
        return *static_cast<AtomStringImpl*>(StringImpl::empty());

    auto firstCharacter = characters.front();
    if (characters.size() == 1) {
        if (firstCharacter <= maxSingleCharacterString)
            return vm().smallStrings.singleCharacterStringRep(firstCharacter);
    } else if (UNLIKELY(characters.size() > maxStringLengthForCache))
        return AtomStringImpl::add(characters).releaseNonNull();

    auto lastCharacter = characters.back();
    auto& slot = cacheSlot(firstCharacter, lastCharacter, characters.size());
    if (UNLIKELY(slot.m_length != characters.size() || !equal(slot.m_buffer, characters))) {
        auto result = AtomStringImpl::add(characters);
        slot.m_impl = result;
        slot.m_length = characters.size();
        WTF::copyElements(slot.m_buffer, characters.data(), characters.size());
        return result.releaseNonNull();
    }

    return *slot.m_impl;
}

ALWAYS_INLINE VM& JSONAtomStringCache::vm() const
{
    return *std::bit_cast<VM*>(std::bit_cast<uintptr_t>(this) - OBJECT_OFFSETOF(VM, jsonAtomStringCache));
}

} // namespace JSC
