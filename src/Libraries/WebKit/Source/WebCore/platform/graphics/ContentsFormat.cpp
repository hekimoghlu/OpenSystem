/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 9, 2022.
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
#include "ContentsFormat.h"

#if USE(CG)
#include "ColorSpaceCG.h"
#endif
#include "DestinationColorSpace.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

std::optional<DestinationColorSpace> contentsFormatExtendedColorSpace(ContentsFormat contentsFormat)
{
    switch (contentsFormat) {
    case ContentsFormat::RGBA8:
        return std::nullopt;
#if HAVE(IOSURFACE_RGB10)
    case ContentsFormat::RGBA10:
        return DestinationColorSpace { extendedSRGBColorSpaceRef() };
#endif
#if HAVE(HDR_SUPPORT)
    case ContentsFormat::RGBA16F:
        return DestinationColorSpace { extendedITUR_2020ColorSpaceRef() };
#endif
    }

    ASSERT_NOT_REACHED();
    return std::nullopt;
}

TextStream& operator<<(TextStream& ts, ContentsFormat contentsFormat)
{
    switch (contentsFormat) {
    case ContentsFormat::RGBA8:
        ts << "RGBA8";
        break;
#if HAVE(IOSURFACE_RGB10)
    case ContentsFormat::RGBA10:
        ts << "RGBA10";
        break;
#endif
#if HAVE(HDR_SUPPORT)
    case ContentsFormat::RGBA16F:
        ts << "RGBA16F";
        break;
#endif
    }
    return ts;
}

} // namespace WebCore
