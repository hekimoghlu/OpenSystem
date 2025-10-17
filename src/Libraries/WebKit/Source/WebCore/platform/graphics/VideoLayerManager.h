/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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

#include "NativeImage.h"
#include "PlatformLayer.h"
#include <wtf/Function.h>

namespace WebCore {

class FloatRect;
class IntSize;

class VideoLayerManager {
public:
    virtual ~VideoLayerManager() = default;

    virtual PlatformLayer* videoInlineLayer() const = 0;
    virtual void setVideoLayer(PlatformLayer*, FloatSize) = 0;
    virtual void didDestroyVideoLayer() = 0;

#if ENABLE(VIDEO_PRESENTATION_MODE)
    virtual PlatformLayer* videoFullscreenLayer() const = 0;
    virtual void setVideoFullscreenLayer(PlatformLayer*, Function<void()>&& completionHandler, PlatformImagePtr) = 0;
    virtual FloatRect videoFullscreenFrame() const = 0;
    virtual void setVideoFullscreenFrame(FloatRect) = 0;
    virtual void updateVideoFullscreenInlineImage(PlatformImagePtr) = 0;
#endif

    virtual void setTextTrackRepresentationLayer(PlatformLayer*) = 0;
    virtual void syncTextTrackBounds() = 0;
};

}
