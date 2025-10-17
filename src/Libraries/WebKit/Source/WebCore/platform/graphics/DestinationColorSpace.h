/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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

#include "PlatformColorSpace.h"
#include <optional>
#include <wtf/Forward.h>

namespace WebCore {

class DestinationColorSpace {
public:
    WEBCORE_EXPORT static const DestinationColorSpace& SRGB();
    WEBCORE_EXPORT static const DestinationColorSpace& LinearSRGB();
#if ENABLE(DESTINATION_COLOR_SPACE_DISPLAY_P3)
    WEBCORE_EXPORT static const DestinationColorSpace& DisplayP3();
#endif

    explicit DestinationColorSpace(PlatformColorSpace platformColorSpace)
        : m_platformColorSpace { WTFMove(platformColorSpace) }
    {
#if USE(CG) || USE(SKIA)
        ASSERT(m_platformColorSpace);
#endif
    }

#if USE(SKIA)
    PlatformColorSpaceValue platformColorSpace() const { return m_platformColorSpace; }
#else
    PlatformColorSpaceValue platformColorSpace() const { return m_platformColorSpace.get(); }
#endif

    PlatformColorSpace serializableColorSpace() const { return m_platformColorSpace; }

    WEBCORE_EXPORT std::optional<DestinationColorSpace> asRGB() const;

    WEBCORE_EXPORT bool supportsOutput() const;

    bool usesExtendedRange() const;

private:
    PlatformColorSpace m_platformColorSpace;
};

WEBCORE_EXPORT bool operator==(const DestinationColorSpace&, const DestinationColorSpace&);

WEBCORE_EXPORT TextStream& operator<<(TextStream&, const DestinationColorSpace&);
}
