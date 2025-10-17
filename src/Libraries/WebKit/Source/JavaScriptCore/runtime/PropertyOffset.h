/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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

#include <wtf/StdLibExtras.h>

namespace JSC {

typedef int PropertyOffset;

static constexpr PropertyOffset invalidOffset = -1;
static constexpr PropertyOffset firstOutOfLineOffset = 64;
static constexpr PropertyOffset knownPolyProtoOffset = 0;
static_assert(knownPolyProtoOffset < firstOutOfLineOffset, "We assume in all the JITs that the poly proto offset is an inline offset");

// Declare all of the functions because they tend to do forward calls.
constexpr inline void checkOffset(PropertyOffset);
constexpr inline void checkOffset(PropertyOffset, int inlineCapacity);
constexpr inline void validateOffset(PropertyOffset);
constexpr inline void validateOffset(PropertyOffset, int inlineCapacity);
constexpr inline bool isValidOffset(PropertyOffset);
constexpr inline bool isInlineOffset(PropertyOffset);
constexpr inline bool isOutOfLineOffset(PropertyOffset);
constexpr inline intptr_t offsetInInlineStorage(PropertyOffset);
constexpr inline intptr_t offsetInOutOfLineStorage(PropertyOffset);
constexpr inline intptr_t offsetInRespectiveStorage(PropertyOffset);
constexpr inline size_t numberOfOutOfLineSlotsForMaxOffset(PropertyOffset);
constexpr inline size_t numberOfSlotsForMaxOffset(PropertyOffset, int inlineCapacity);

constexpr inline void checkOffset(PropertyOffset offset)
{
    UNUSED_PARAM(offset);
    ASSERT(offset >= invalidOffset);
}

constexpr inline void checkOffset(PropertyOffset offset, int inlineCapacity)
{
    UNUSED_PARAM(offset);
    UNUSED_PARAM(inlineCapacity);
    ASSERT(offset >= invalidOffset);
    ASSERT(offset == invalidOffset
        || offset < inlineCapacity
        || isOutOfLineOffset(offset));
}

constexpr inline void validateOffset(PropertyOffset offset)
{
    checkOffset(offset);
    ASSERT(isValidOffset(offset));
}

constexpr inline void validateOffset(PropertyOffset offset, int inlineCapacity)
{
    checkOffset(offset, inlineCapacity);
    ASSERT(isValidOffset(offset));
}

constexpr inline bool isValidOffset(PropertyOffset offset)
{
    checkOffset(offset);
    return offset != invalidOffset;
}

constexpr inline bool isInlineOffset(PropertyOffset offset)
{
    checkOffset(offset);
    return offset < firstOutOfLineOffset;
}

constexpr inline bool isOutOfLineOffset(PropertyOffset offset)
{
    checkOffset(offset);
    return !isInlineOffset(offset);
}

constexpr inline intptr_t offsetInInlineStorage(PropertyOffset offset)
{
    validateOffset(offset);
    ASSERT(isInlineOffset(offset));
    return offset;
}

constexpr inline intptr_t offsetInOutOfLineStorage(PropertyOffset offset)
{
    validateOffset(offset);
    ASSERT(isOutOfLineOffset(offset));
    return -static_cast<ptrdiff_t>(offset - firstOutOfLineOffset) - 1;
}

constexpr inline intptr_t offsetInRespectiveStorage(PropertyOffset offset)
{
    if (isInlineOffset(offset))
        return offsetInInlineStorage(offset);
    return offsetInOutOfLineStorage(offset);
}

constexpr inline size_t numberOfOutOfLineSlotsForMaxOffset(PropertyOffset offset)
{
    checkOffset(offset);
    if (offset < firstOutOfLineOffset)
        return 0;
    return offset - firstOutOfLineOffset + 1;
}

constexpr inline size_t numberOfSlotsForMaxOffset(PropertyOffset offset, int inlineCapacity)
{
    checkOffset(offset, inlineCapacity);
    if (offset < inlineCapacity)
        return offset + 1;
    return inlineCapacity + numberOfOutOfLineSlotsForMaxOffset(offset);
}

constexpr inline PropertyOffset offsetForPropertyNumber(int propertyNumber, int inlineCapacity)
{
    PropertyOffset offset = propertyNumber;
    if (offset >= inlineCapacity) {
        offset += firstOutOfLineOffset;
        offset -= inlineCapacity;
    }
    return offset;
}

} // namespace JSC
