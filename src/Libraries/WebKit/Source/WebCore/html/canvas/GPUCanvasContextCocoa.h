/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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

#include "GPU.h"
#include "GPUBasedCanvasRenderingContext.h"
#include "GPUCanvasConfiguration.h"
#include "GPUCanvasContext.h"
#include "GPUPresentationContext.h"
#include "GPUTexture.h"
#include "GraphicsLayerContentsDisplayDelegate.h"
#include "HTMLCanvasElement.h"
#include "IOSurface.h"
#include "OffscreenCanvas.h"
#include "PlatformCALayer.h"
#include <variant>
#include <wtf/MachSendRight.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class GPUDisplayBufferDisplayDelegate;

class GPUCanvasContextCocoa final : public GPUCanvasContext {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(GPUCanvasContextCocoa);
public:
#if ENABLE(OFFSCREEN_CANVAS)
    using CanvasType = std::variant<RefPtr<HTMLCanvasElement>, RefPtr<OffscreenCanvas>>;
#else
    using CanvasType = std::variant<RefPtr<HTMLCanvasElement>>;
#endif

    static std::unique_ptr<GPUCanvasContextCocoa> create(CanvasBase&, GPU&);

    DestinationColorSpace colorSpace() const override;
    bool compositingResultsNeedUpdating() const override { return m_compositingResultsNeedsUpdating; }
    RefPtr<GraphicsLayerContentsDisplayDelegate> layerContentsDisplayDelegate() override;
    bool needsPreparationForDisplay() const override { return true; }
    void prepareForDisplay() override;
    ImageBufferPixelFormat pixelFormat() const override;
    void reshape() override;


    RefPtr<ImageBuffer> surfaceBufferToImageBuffer(SurfaceBuffer) override;
    // GPUCanvasContext methods:
    CanvasType canvas() override;
    ExceptionOr<void> configure(GPUCanvasConfiguration&&) override;
    void unconfigure() override;
    std::optional<GPUCanvasConfiguration> getConfiguration() const override;
    ExceptionOr<RefPtr<GPUTexture>> getCurrentTexture() override;
    RefPtr<ImageBuffer> transferToImageBuffer() override;

private:
    explicit GPUCanvasContextCocoa(CanvasBase&, Ref<GPUCompositorIntegration>&&, Ref<GPUPresentationContext>&&);

    void markContextChangedAndNotifyCanvasObservers();

    bool isConfigured() const
    {
        return static_cast<bool>(m_configuration);
    }

    CanvasType htmlOrOffscreenCanvas() const;
    ExceptionOr<void> configure(GPUCanvasConfiguration&&, bool);
    void present(uint32_t frameIndex);

    struct Configuration {
        Ref<GPUDevice> device;
        GPUTextureFormat format { GPUTextureFormat::R8unorm };
        GPUTextureUsageFlags usage { GPUTextureUsage::RENDER_ATTACHMENT };
        Vector<GPUTextureFormat> viewFormats;
        GPUPredefinedColorSpace colorSpace { GPUPredefinedColorSpace::SRGB };
        GPUCanvasToneMapping toneMapping;
        GPUCanvasAlphaMode compositingAlphaMode { GPUCanvasAlphaMode::Opaque };
        Vector<MachSendRight> renderBuffers;
        unsigned frameCount { 0 };
    };
    std::optional<Configuration> m_configuration;

    const Ref<GPUDisplayBufferDisplayDelegate> m_layerContentsDisplayDelegate;
    const Ref<GPUCompositorIntegration> m_compositorIntegration;
    const Ref<GPUPresentationContext> m_presentationContext;
    RefPtr<GPUTexture> m_currentTexture;

    GPUIntegerCoordinate m_width { 0 };
    GPUIntegerCoordinate m_height { 0 };
    bool m_compositingResultsNeedsUpdating { false };
};

} // namespace WebCore
