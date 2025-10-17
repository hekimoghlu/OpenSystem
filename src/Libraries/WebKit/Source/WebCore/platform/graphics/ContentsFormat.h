/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 30, 2021.
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

#include <optional>

namespace WTF {
class TextStream;
}

namespace WebCore {

class DestinationColorSpace;

enum class ContentsFormat : uint8_t {
    RGBA8,
#if HAVE(IOSURFACE_RGB10)
    RGBA10,
#endif
#if HAVE(HDR_SUPPORT)
    RGBA16F,
#endif
};

constexpr unsigned contentsFormatBytesPerPixel(ContentsFormat contentsFormat, bool isOpaque)
{
#if !HAVE(IOSURFACE_RGB10)
    UNUSED_PARAM(isOpaque);
#endif

    switch (contentsFormat) {
    case ContentsFormat::RGBA8:
        return 4;
#if HAVE(IOSURFACE_RGB10)
    case ContentsFormat::RGBA10:
        return isOpaque ? 4 : 5;
#endif
#if HAVE(HDR_SUPPORT)
    case ContentsFormat::RGBA16F:
        return 8;
#endif
    }

    ASSERT_NOT_REACHED();
    return 4;
}

WEBCORE_EXPORT std::optional<DestinationColorSpace> contentsFormatExtendedColorSpace(ContentsFormat);
WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, ContentsFormat);

} // namespace WebCore
