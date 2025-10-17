/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 5, 2024.
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

#include <unicode/utext.h>

namespace WTF {

enum class UTextProviderContext {
    NoContext,
    PriorContext,
    PrimaryContext
};

inline UTextProviderContext uTextProviderContext(const UText* text, int64_t nativeIndex, UBool forward)
{
    if (!text->b || nativeIndex > text->b)
        return UTextProviderContext::PrimaryContext;
    if (nativeIndex == text->b)
        return forward ? UTextProviderContext::PrimaryContext : UTextProviderContext::PriorContext;
    return UTextProviderContext::PriorContext;
}

template<typename CharacterType>
inline void initializeContextAwareUTextProvider(UText* text, const UTextFuncs* funcs, std::span<const CharacterType> string, std::span<const UChar> priorContext)
{
    text->pFuncs = funcs;
    text->providerProperties = 1 << UTEXT_PROVIDER_STABLE_CHUNKS;
    text->context = string.data();
    text->p = string.data();
    text->a = string.size();
    text->q = priorContext.data();
    text->b = priorContext.size();
}

// Shared implementation for the UTextClone function on UTextFuncs.

UText* uTextCloneImpl(UText* destination, const UText* source, UBool deep, UErrorCode* status);


// Helpers for the UTextAccess function on UTextFuncs.

inline int64_t uTextAccessPinIndex(int64_t& index, int64_t limit)
{
    if (index < 0)
        index = 0;
    else if (index > limit)
        index = limit;
    return index;
}

inline bool uTextAccessInChunkOrOutOfRange(UText* text, int64_t nativeIndex, int64_t nativeLength, UBool forward, UBool& isAccessible)
{
    if (forward) {
        if (nativeIndex >= text->chunkNativeStart && nativeIndex < text->chunkNativeLimit) {
            int64_t offset = nativeIndex - text->chunkNativeStart;
            // Ensure chunk offset is well formed if computed offset exceeds int32_t range.
            ASSERT(offset < std::numeric_limits<int32_t>::max());
            text->chunkOffset = offset < std::numeric_limits<int32_t>::max() ? static_cast<int32_t>(offset) : 0;
            isAccessible = true;
            return true;
        }
        if (nativeIndex >= nativeLength && text->chunkNativeLimit == nativeLength) {
            text->chunkOffset = text->chunkLength;
            isAccessible = false;
            return true;
        }
    } else {
        if (nativeIndex > text->chunkNativeStart && nativeIndex <= text->chunkNativeLimit) {
            int64_t offset = nativeIndex - text->chunkNativeStart;
            // Ensure chunk offset is well formed if computed offset exceeds int32_t range.
            ASSERT(offset < std::numeric_limits<int32_t>::max());
            text->chunkOffset = offset < std::numeric_limits<int32_t>::max() ? static_cast<int32_t>(offset) : 0;
            isAccessible = true;
            return true;
        }
        if (nativeIndex <= 0 && !text->chunkNativeStart) {
            text->chunkOffset = 0;
            isAccessible = false;
            return true;
        }
    }
    return false;
}

} // namespace WTF
