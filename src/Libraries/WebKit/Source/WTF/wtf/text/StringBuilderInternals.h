/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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

#include <wtf/text/StringBuilder.h>

namespace WTF {

// Allocate a new buffer, copying in currentCharacters (these may come from either m_string or m_buffer.
template<typename AllocationCharacterType, typename CurrentCharacterType> void StringBuilder::allocateBuffer(std::span<const CurrentCharacterType> currentCharactersToCopy, unsigned requiredCapacity)
{
    std::span<AllocationCharacterType> newBufferCharacters;
    auto buffer = StringImpl::tryCreateUninitialized(requiredCapacity, newBufferCharacters);
    if (UNLIKELY(!buffer)) {
        didOverflow();
        return;
    }

    ASSERT(!hasOverflowed());
    StringImpl::copyCharacters(newBufferCharacters, currentCharactersToCopy);

    m_buffer = WTFMove(buffer);
    m_string = { };
}

template<typename CharacterType> void StringBuilder::reallocateBuffer(unsigned requiredCapacity)
{
    // If the buffer has only one ref (by this StringBuilder), reallocate it.
    if (m_buffer) {
        m_string = { }; // Clear the string to remove the reference to m_buffer if any before checking the reference count of m_buffer.
        if (m_buffer->hasOneRef()) {
            CharacterType* bufferCharacters;
            auto buffer = StringImpl::tryReallocate(m_buffer.releaseNonNull(), requiredCapacity, bufferCharacters);
            if (UNLIKELY(!buffer)) {
                didOverflow();
                return;
            }
            m_buffer = WTFMove(*buffer);
            return;
        }
    }

    allocateBuffer<CharacterType>(span<CharacterType>(), requiredCapacity);
}

// Make 'additionalLength' additional capacity be available in m_buffer, update m_string & m_length to use,
// that capacity and return a pointer to the newly allocated storage so the caller can write characters there.
// Returns nullptr if allocation fails, length overflows, or if total capacity is 0 so no buffer is needed.
// The caller has the responsibility for checking that CharacterType is the type of the existing buffer.
template<typename CharacterType> std::span<CharacterType> StringBuilder::extendBufferForAppending(unsigned requiredLength)
{
    if (m_buffer && requiredLength <= m_buffer->length()) {
        m_string = { };
        return spanConstCast<CharacterType>(m_buffer->span<CharacterType>().subspan(std::exchange(m_length, requiredLength)));
    }
    return extendBufferForAppendingSlowCase<CharacterType>(requiredLength);
}

// Shared by the other extendBuffer functions.
template<typename CharacterType> std::span<CharacterType> StringBuilder::extendBufferForAppendingSlowCase(unsigned requiredLength)
{
    if (!requiredLength || hasOverflowed())
        return { };
    reallocateBuffer(expandedCapacity(capacity(), requiredLength));
    if (UNLIKELY(hasOverflowed()))
        return { };
    return spanConstCast<CharacterType>(m_buffer->span<CharacterType>().subspan(std::exchange(m_length, requiredLength)));
}

} // namespace WTF
