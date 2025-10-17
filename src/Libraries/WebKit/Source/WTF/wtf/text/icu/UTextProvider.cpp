/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 9, 2022.
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
#include "config.h"
#include <wtf/text/icu/UTextProvider.h>

#include <algorithm>
#include <string.h>

namespace WTF {

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
// Relocate pointer from source into destination as required.
static inline void fixPointer(const UText* source, UText* destination, const void*& pointer)
{
    if (pointer >= source->pExtra && pointer < static_cast<char*>(source->pExtra) + source->extraSize) {
        // Pointer references source extra buffer.
        pointer = static_cast<char*>(destination->pExtra) + (static_cast<const char*>(pointer) - static_cast<const char*>(source->pExtra));
    } else if (pointer >= source && pointer < reinterpret_cast<const char*>(source) + source->sizeOfStruct) {
        // Pointer references source text structure, but not source extra buffer.
        pointer = reinterpret_cast<char*>(destination) + (static_cast<const char*>(pointer) - reinterpret_cast<const char*>(source));
    }
}
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

UText* uTextCloneImpl(UText* destination, const UText* source, UBool deep, UErrorCode* status)
{
    ASSERT_UNUSED(deep, !deep);
    if (U_FAILURE(*status))
        return nullptr;
    int32_t extraSize = source->extraSize;
    destination = utext_setup(destination, extraSize, status);
    if (U_FAILURE(*status))
        return destination;
    void* extraNew = destination->pExtra;
    int32_t flags = destination->flags;
    int sizeToCopy = std::min(source->sizeOfStruct, destination->sizeOfStruct);
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    memcpy(destination, source, sizeToCopy);
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    destination->pExtra = extraNew;
    destination->flags = flags;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    memcpy(destination->pExtra, source->pExtra, extraSize);
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    fixPointer(source, destination, destination->context);
    fixPointer(source, destination, destination->p);
    fixPointer(source, destination, destination->q);
    ASSERT(!destination->r);
    const void* chunkContents = static_cast<const void*>(destination->chunkContents);
    fixPointer(source, destination, chunkContents);
    destination->chunkContents = static_cast<const UChar*>(chunkContents);
    return destination;
}

} // namespace WTF
