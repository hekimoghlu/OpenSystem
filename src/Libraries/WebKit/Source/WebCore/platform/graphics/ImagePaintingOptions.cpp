/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 4, 2024.
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
#include "ImagePaintingOptions.h"

#include <wtf/text/TextStream.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

TextStream& operator<<(TextStream& ts, ImagePaintingOptions options)
{
    ts.dumpProperty("composite-operator", options.compositeOperator());
    ts.dumpProperty("blend-mode", options.blendMode());
    ts.dumpProperty("decoding-mode", options.decodingMode());
    ts.dumpProperty("orientation", options.orientation().orientation());
    ts.dumpProperty("interpolation-quality", options.interpolationQuality());
    return ts;
}

TextStream& operator<<(TextStream& ts, DecodingMode mode)
{
    switch (mode) {
    case DecodingMode::Auto:
        ts << "auto";
        break;
    case DecodingMode::Synchronous:
        ts << "synchronous";
        break;
    case DecodingMode::Asynchronous:
        ts << "asynchronous";
        break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, ImageOrientation::Orientation orientation)
{
    using Orientation = ImageOrientation::Orientation;
    switch (orientation) {
    case Orientation::FromImage:
        ts << "from-image";
        break;
    case Orientation::OriginTopLeft:
        ts << "origin-top-left";
        break;
    case Orientation::OriginTopRight:
        ts << "origin-bottom-right";
        break;
    case Orientation::OriginBottomRight:
        ts << "origin-top-right";
        break;
    case Orientation::OriginBottomLeft:
        ts << "origin-top-left";
        break;
    case Orientation::OriginLeftTop:
        ts << "origin-left-bottom";
        break;
    case Orientation::OriginRightTop:
        ts << "origin-right-bottom";
        break;
    case Orientation::OriginRightBottom:
        ts << "origin-right-top";
        break;
    case Orientation::OriginLeftBottom:
        ts << "origin-left-top";
        break;
    }
    return ts;
}

}
