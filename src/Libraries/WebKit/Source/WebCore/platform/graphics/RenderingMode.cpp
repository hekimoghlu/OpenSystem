/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 24, 2023.
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
#include "RenderingMode.h"

#include <wtf/text/TextStream.h>

namespace WebCore {

TextStream& operator<<(TextStream& ts, RenderingPurpose purpose)
{
    switch (purpose) {
    case RenderingPurpose::Unspecified: ts << "Unspecified"; break;
    case RenderingPurpose::Canvas: ts << "Canvas"; break;
    case RenderingPurpose::DOM: ts << "DOM"; break;
    case RenderingPurpose::LayerBacking: ts << "LayerBacking"; break;
    case RenderingPurpose::BitmapOnlyLayerBacking: ts << "BitmapOnlyLayerBacking"; break;
    case RenderingPurpose::Snapshot: ts << "Snapshot"; break;
    case RenderingPurpose::ShareableSnapshot: ts << "ShareableSnapshot"; break;
    case RenderingPurpose::ShareableLocalSnapshot: ts << "ShareableLocalSnapshot"; break;
    case RenderingPurpose::MediaPainting: ts << "MediaPainting"; break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, RenderingMode mode)
{
    switch (mode) {
    case RenderingMode::Unaccelerated: ts << "Unaccelerated"; break;
    case RenderingMode::Accelerated: ts << "Accelerated"; break;
    case RenderingMode::PDFDocument: ts << "PDFDocument"; break;
    case RenderingMode::DisplayList: ts << "DisplayList"; break;
    }
    return ts;
}

TextStream& operator<<(TextStream& ts, RenderingMethod method)
{
    switch (method) {
    case RenderingMethod::Local: ts << "Local"; break;
    }

    return ts;
}

} // namespace WebCore
