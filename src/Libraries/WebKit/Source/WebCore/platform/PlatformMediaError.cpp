/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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
#include "PlatformMediaError.h"

#include <wtf/NeverDestroyed.h>

namespace WebCore {

String convertEnumerationToString(PlatformMediaError enumerationValue)
{
    static std::array<const NeverDestroyed<String>, 13> values {
        MAKE_STATIC_STRING_IMPL("AppendError"),
        MAKE_STATIC_STRING_IMPL("ClientDisconnected"),
        MAKE_STATIC_STRING_IMPL("BufferRemoved"),
        MAKE_STATIC_STRING_IMPL("SourceRemoved"),
        MAKE_STATIC_STRING_IMPL("IPCError"),
        MAKE_STATIC_STRING_IMPL("ParsingError"),
        MAKE_STATIC_STRING_IMPL("MemoryError"),
        MAKE_STATIC_STRING_IMPL("Cancelled"),
        MAKE_STATIC_STRING_IMPL("LogicError"),
        MAKE_STATIC_STRING_IMPL("DecoderCreationError"),
        MAKE_STATIC_STRING_IMPL("NotSupportedError"),
        MAKE_STATIC_STRING_IMPL("NetworkError"),
        MAKE_STATIC_STRING_IMPL("NotReady"),
    };
    static_assert(!static_cast<size_t>(PlatformMediaError::AppendError), "PlatformMediaError::AppendError is not 0 as expected");
    static_assert(static_cast<size_t>(PlatformMediaError::ClientDisconnected) == 1, "PlatformMediaError::ClientDisconnected is not 1 as expected");
    static_assert(static_cast<size_t>(PlatformMediaError::BufferRemoved) == 2, "PlatformMediaError::BufferRemoved is not 2 as expected");
    static_assert(static_cast<size_t>(PlatformMediaError::SourceRemoved) == 3, "PlatformMediaError::SourceRemoved is not 3 as expected");
    static_assert(static_cast<size_t>(PlatformMediaError::IPCError) == 4, "PlatformMediaError::IPCError is not 4 as expected");
    static_assert(static_cast<size_t>(PlatformMediaError::ParsingError) == 5, "PlatformMediaError::ParsingError is not 5 as expected");
    static_assert(static_cast<size_t>(PlatformMediaError::MemoryError) == 6, "PlatformMediaError::MemoryError is not 6 as expected");
    static_assert(static_cast<size_t>(PlatformMediaError::Cancelled) == 7, "PlatformMediaError::Cancelled is not 7 as expected");
    static_assert(static_cast<size_t>(PlatformMediaError::LogicError) == 8, "PlatformMediaError::LogicError is not 8 as expected");
    static_assert(static_cast<size_t>(PlatformMediaError::DecoderCreationError) == 9, "PlatformMediaError::DecoderCreationError is not 9 as expected");
    static_assert(static_cast<size_t>(PlatformMediaError::NotSupportedError) == 10, "PlatformMediaError::NotSupportedError is not 10 as expected");
    static_assert(static_cast<size_t>(PlatformMediaError::NetworkError) == 11, "PlatformMediaError::NetworkError is not 11 as expected");
    static_assert(static_cast<size_t>(PlatformMediaError::NotReady) == 12, "PlatformMediaError::NotReady is not 12 as expected");
    ASSERT(static_cast<size_t>(enumerationValue) < std::size(values));
    return values[static_cast<size_t>(enumerationValue)];
}

} // namespace WebCore
