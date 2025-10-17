/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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

#include "GPUOrigin2DDict.h"
#include "HTMLCanvasElement.h"
#include "HTMLImageElement.h"
#include "HTMLVideoElement.h"
#include "ImageBitmap.h"
#include "ImageData.h"
#include "OffscreenCanvas.h"
#include "WebCodecsVideoFrame.h"
#include "WebGPUImageCopyExternalImage.h"
#include <optional>
#include <variant>
#include <wtf/RefPtr.h>

namespace WebCore {

struct GPUImageCopyExternalImage {
    using SourceType = std::variant<RefPtr<ImageBitmap>,
#if ENABLE(VIDEO) && ENABLE(WEB_CODECS)
    RefPtr<ImageData>, RefPtr<HTMLImageElement>, RefPtr<HTMLVideoElement>, RefPtr<WebCodecsVideoFrame>,
#endif
#if ENABLE(OFFSCREEN_CANVAS)
    RefPtr<OffscreenCanvas>,
#endif
    RefPtr<HTMLCanvasElement>>;

    WebGPU::ImageCopyExternalImage convertToBacking() const
    {
        return {
            // FIXME: Handle the canvas element.
            origin ? std::optional { WebCore::convertToBacking(*origin) } : std::nullopt,
            flipY,
        };
    }

    SourceType source;
    std::optional<GPUOrigin2D> origin;
    bool flipY { false };
};

}
