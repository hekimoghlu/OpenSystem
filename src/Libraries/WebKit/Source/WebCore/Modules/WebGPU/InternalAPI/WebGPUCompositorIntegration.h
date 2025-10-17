/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 11, 2025.
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

#include "AlphaPremultiplication.h"

#include <optional>
#include <wtf/CompletionHandler.h>
#include <wtf/Function.h>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(COCOA)
#include <wtf/MachSendRight.h>
#include <wtf/Vector.h>
#endif

namespace WebCore {
class DestinationColorSpace;
class ImageBuffer;
class NativeImage;
}

namespace WebCore::WebGPU {

class Device;

enum class TextureFormat : uint8_t;

class CompositorIntegration : public RefCountedAndCanMakeWeakPtr<CompositorIntegration> {
public:
    virtual ~CompositorIntegration() = default;

#if PLATFORM(COCOA)
    virtual Vector<MachSendRight> recreateRenderBuffers(int width, int height, WebCore::DestinationColorSpace&&, WebCore::AlphaPremultiplication, WebCore::WebGPU::TextureFormat, Device&) = 0;
#endif

    virtual void prepareForDisplay(uint32_t frameIndex, CompletionHandler<void()>&&) = 0;
    virtual void withDisplayBufferAsNativeImage(uint32_t bufferIndex, Function<void(WebCore::NativeImage*)>) = 0;
    virtual void paintCompositedResultsToCanvas(WebCore::ImageBuffer&, uint32_t bufferIndex) = 0;

protected:
    CompositorIntegration() = default;

private:
    CompositorIntegration(const CompositorIntegration&) = delete;
    CompositorIntegration(CompositorIntegration&&) = delete;
    CompositorIntegration& operator=(const CompositorIntegration&) = delete;
    CompositorIntegration& operator=(CompositorIntegration&&) = delete;
};

} // namespace WebCore::WebGPU
