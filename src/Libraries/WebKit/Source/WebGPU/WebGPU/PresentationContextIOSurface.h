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
#pragma once

#import "PresentationContext.h"
#import <wtf/MachSendRight.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/Vector.h>
#import <wtf/spi/cocoa/IOSurfaceSPI.h>

namespace WebGPU {

class Device;
class Instance;

class PresentationContextIOSurface : public PresentationContext {
    WTF_MAKE_TZONE_ALLOCATED(PresentationContextIOSurface);
public:
    static Ref<PresentationContextIOSurface> create(const WGPUSurfaceDescriptor&, const Instance&);

    virtual ~PresentationContextIOSurface();

    void configure(Device&, const WGPUSwapChainDescriptor&) override;
    void unconfigure() override;

    void present(uint32_t) override;
    Texture* getCurrentTexture(uint32_t) override;
    TextureView* getCurrentTextureView() override;

    bool isPresentationContextIOSurface() const override { return true; }

    bool isValid() override { return true; }
private:
    PresentationContextIOSurface(const WGPUSurfaceDescriptor&, const Instance&);

    void renderBuffersWereRecreated(NSArray<IOSurface *> *renderBuffers);
    void onSubmittedWorkScheduled(Function<void()>&&);
    RetainPtr<CGImageRef> getTextureAsNativeImage(uint32_t bufferIndex, bool& isIOSurfaceSupportedFormat) final;

    NSArray<IOSurface *> *m_ioSurfaces { nil };
    struct RenderBuffer {
        Ref<Texture> texture;
        RefPtr<Texture> luminanceClampTexture;
    };
    Vector<RenderBuffer> m_renderBuffers;
    RefPtr<Device> m_device;
    RefPtr<Texture> m_invalidTexture;
    id<MTLFunction> m_luminanceClampFunction;
    id<MTLComputePipelineState> m_computePipelineState;
#if HAVE(IOSURFACE_SET_OWNERSHIP_IDENTITY) && HAVE(TASK_IDENTITY_TOKEN)
    std::optional<const MachSendRight> m_webProcessID;
#endif
    WGPUColorSpace m_colorSpace { WGPUColorSpace::SRGB };
    WGPUToneMappingMode m_toneMappingMode { WGPUToneMappingMode_Standard };
    WGPUCompositeAlphaMode m_alphaMode { WGPUCompositeAlphaMode_Premultiplied };
};

} // namespace WebGPU

SPECIALIZE_TYPE_TRAITS_WEBGPU_PRESENTATION_CONTEXT(PresentationContextIOSurface, isPresentationContextIOSurface());
