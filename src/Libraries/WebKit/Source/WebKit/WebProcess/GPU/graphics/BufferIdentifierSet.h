/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 9, 2025.
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

#if ENABLE(GPU_PROCESS)

#include <WebCore/ProcessQualified.h>
#include <WebCore/RenderingResourceIdentifier.h>

namespace WebKit {

struct BufferIdentifierSet {
    std::optional<WebCore::RenderingResourceIdentifier> front;
    std::optional<WebCore::RenderingResourceIdentifier> back;
    std::optional<WebCore::RenderingResourceIdentifier> secondaryBack;
};

inline TextStream& operator<<(TextStream& ts, const BufferIdentifierSet& set)
{
    auto dumpBuffer = [&](const char* name, const std::optional<WebCore::RenderingResourceIdentifier>& bufferInfo) {
        ts.startGroup();
        ts << name << " ";
        if (bufferInfo)
            ts << *bufferInfo;
        else
            ts << "none";
        ts.endGroup();
    };
    dumpBuffer("front buffer", set.front);
    dumpBuffer("back buffer", set.back);
    dumpBuffer("secondaryBack buffer", set.secondaryBack);

    return ts;
}

enum class BufferInSetType : uint8_t {
    Front = 1 << 0,
    Back = 1 << 1,
    SecondaryBack = 1 << 2,
};

inline TextStream& operator<<(TextStream& ts, BufferInSetType bufferType)
{
    if (bufferType == BufferInSetType::Front)
        ts << "Front";
    else if (bufferType == BufferInSetType::Back)
        ts << "Back";
    else
        ts << "SecondaryBack";

    return ts;
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
