/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 10, 2022.
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

#import "PresentationContext.h"

#import <QuartzCore/QuartzCore.h>
#import <optional>
#import <wtf/TZoneMalloc.h>
#import <wtf/text/WTFString.h>

namespace WebGPU {

class PresentationContextCoreAnimation : public PresentationContext {
    WTF_MAKE_TZONE_ALLOCATED(PresentationContextCoreAnimation);
public:
    static Ref<PresentationContextCoreAnimation> create(const WGPUSurfaceDescriptor& descriptor)
    {
        return adoptRef(*new PresentationContextCoreAnimation(descriptor));
    }

    virtual ~PresentationContextCoreAnimation();

    void configure(Device&, const WGPUSwapChainDescriptor&) override;
    void unconfigure() override;

    void present(uint32_t) override;
    Texture* getCurrentTexture(uint32_t) override;
    TextureView* getCurrentTextureView() override;

    bool isPresentationContextCoreAnimation() const override { return true; }

    bool isValid() override { return m_configuration != std::nullopt; }

private:
    PresentationContextCoreAnimation(const WGPUSurfaceDescriptor&);

    struct Configuration {
        Configuration(uint32_t width, uint32_t height, WGPUTextureUsageFlags usage, String&& label, WGPUTextureFormat format, Device& device)
            : width(width)
            , height(height)
            , usage(usage)
            , label(WTFMove(label))
            , format(format)
            , device(device)
        {
        }

        struct FrameState {
            id<CAMetalDrawable> currentDrawable;
            RefPtr<Texture> currentTexture;
            RefPtr<TextureView> currentTextureView;
        };

        FrameState generateCurrentFrameState(CAMetalLayer *);

        std::optional<FrameState> currentFrameState;
        uint32_t width { 0 };
        uint32_t height { 0 };
        WGPUTextureUsageFlags usage { WGPUTextureUsage_None };
        String label;
        WGPUTextureFormat format { WGPUTextureFormat_Undefined };
        Ref<Device> device;
    };


    CAMetalLayer *m_layer { nil };
    std::optional<Configuration> m_configuration;
};

} // namespace WebGPU

SPECIALIZE_TYPE_TRAITS_WEBGPU_PRESENTATION_CONTEXT(PresentationContextCoreAnimation, isPresentationContextCoreAnimation());
