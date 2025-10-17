/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 27, 2024.
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
#include "ContentExtensionStringSerialization.h"

#if ENABLE(CONTENT_EXTENSIONS)

#include <wtf/StdLibExtras.h>

namespace WebCore::ContentExtensions {

String deserializeString(std::span<const uint8_t> span)
{
    auto serializedLength = stringSerializedLength(span);
    return String::fromUTF8(span.subspan(sizeof(uint32_t), serializedLength - sizeof(uint32_t)));
}

void serializeString(Vector<uint8_t>& actions, const String& string)
{
    auto utf8 = string.utf8();
    uint32_t serializedLength = sizeof(uint32_t) + utf8.length();
    actions.reserveCapacity(actions.size() + serializedLength);
    actions.append(asByteSpan(serializedLength));
    actions.append(utf8.span());
}

size_t stringSerializedLength(std::span<const uint8_t> span)
{
    return reinterpretCastSpanStartTo<const uint32_t>(span);
}

} // namespace WebCore::ContentExtensions

#endif // ENABLE(CONTENT_EXTENSIONS)
