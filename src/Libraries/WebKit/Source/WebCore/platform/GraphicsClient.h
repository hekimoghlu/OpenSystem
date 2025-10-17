/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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

#include "PlatformScreen.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class DestinationColorSpace;
class GraphicsContextGL;
class ImageBuffer;
class SerializedImageBuffer;

struct GraphicsContextGLAttributes;

enum class ImageBufferPixelFormat : uint8_t;
enum class RenderingMode : uint8_t;
enum class RenderingPurpose : uint8_t;

namespace WebGPU {
class GPU;
}

class GraphicsClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(GraphicsClient);
    WTF_MAKE_NONCOPYABLE(GraphicsClient);
public:
    GraphicsClient() = default;
    virtual ~GraphicsClient() = default;

    virtual PlatformDisplayID displayID() const = 0;

#if ENABLE(WEBGL)
    virtual RefPtr<GraphicsContextGL> createGraphicsContextGL(const GraphicsContextGLAttributes&) const = 0;
#endif
#if HAVE(WEBGPU_IMPLEMENTATION)
    virtual RefPtr<WebCore::WebGPU::GPU> createGPUForWebGPU() const = 0;
#endif

private:
    // Called by passing GraphicsClient into ImageBuffer functions.
    virtual RefPtr<ImageBuffer> createImageBuffer(const FloatSize&, RenderingMode, RenderingPurpose, float resolutionScale, const DestinationColorSpace&, ImageBufferPixelFormat) const = 0;

    // Called by passing GraphicsClient into SerializedImageBuffer functions.
    virtual RefPtr<WebCore::ImageBuffer> sinkIntoImageBuffer(std::unique_ptr<WebCore::SerializedImageBuffer>) = 0;

    friend class ImageBuffer;
    friend class SerializedImageBuffer;
};

} // namespace WebCore
