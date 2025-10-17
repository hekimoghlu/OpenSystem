/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 7, 2022.
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
#import "config.h"
#import "PresentationContext.h"

#import "APIConversions.h"
#import "Adapter.h"
#import "PresentationContextCoreAnimation.h"
#import "PresentationContextIOSurface.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebGPU {

Ref<PresentationContext> Device::createSwapChain(PresentationContext& presentationContext, const WGPUSwapChainDescriptor& descriptor)
{
    presentationContext.configure(*this, descriptor);
    return presentationContext;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(PresentationContext);

Ref<PresentationContext> PresentationContext::create(const WGPUSurfaceDescriptor& descriptor, const Instance& instance)
{
    if (!descriptor.nextInChain || descriptor.nextInChain->next)
        return PresentationContext::createInvalid();

    switch (static_cast<unsigned>(descriptor.nextInChain->sType)) {
    case WGPUSTypeExtended_SurfaceDescriptorCocoaSurfaceBacking:
        return PresentationContextIOSurface::create(descriptor, instance);
    case WGPUSType_SurfaceDescriptorFromMetalLayer:
        return PresentationContextCoreAnimation::create(descriptor);
    default:
        return PresentationContext::createInvalid();
    }
}

PresentationContext::PresentationContext() = default;

PresentationContext::~PresentationContext() = default;

WGPUTextureFormat PresentationContext::getPreferredFormat(const Adapter&)
{
    return WGPUTextureFormat_BGRA8Unorm;
}

void PresentationContext::configure(Device&, const WGPUSwapChainDescriptor&)
{
}

void PresentationContext::unconfigure()
{
}

void PresentationContext::present(uint32_t)
{
}

Texture* PresentationContext::getCurrentTexture(uint32_t)
{
    return nullptr;
}

TextureView* PresentationContext::getCurrentTextureView()
{
    return nullptr;
}

} // namespace WebGPU

#pragma mark WGPU Stubs

void wgpuSurfaceReference(WGPUSurface surface)
{
    WebGPU::fromAPI(surface).ref();
}

void wgpuSurfaceRelease(WGPUSurface surface)
{
    WebGPU::fromAPI(surface).deref();
}

void wgpuSwapChainReference(WGPUSwapChain swapChain)
{
    WebGPU::fromAPI(swapChain).ref();
}

void wgpuSwapChainRelease(WGPUSwapChain swapChain)
{
    WebGPU::fromAPI(swapChain).deref();
}

WGPUTextureFormat wgpuSurfaceGetPreferredFormat(WGPUSurface surface, WGPUAdapter adapter)
{
    return WebGPU::protectedFromAPI(surface)->getPreferredFormat(WebGPU::protectedFromAPI(adapter));
}

WGPUTexture wgpuSwapChainGetCurrentTexture(WGPUSwapChain swapChain, uint32_t index)
{
    return WebGPU::protectedFromAPI(swapChain)->getCurrentTexture(index);
}

WGPUTextureView wgpuSwapChainGetCurrentTextureView(WGPUSwapChain swapChain)
{
    return WebGPU::protectedFromAPI(swapChain)->getCurrentTextureView();
}

void wgpuSwapChainPresent(WGPUSwapChain swapChain, uint32_t index)
{
    WebGPU::protectedFromAPI(swapChain)->present(index);
}

RetainPtr<CGImageRef> wgpuSwapChainGetTextureAsNativeImage(WGPUSwapChain swapChain, uint32_t bufferIndex, bool& isIOSurfaceSupportedFormat)
{
    return WebGPU::protectedFromAPI(swapChain)->getTextureAsNativeImage(bufferIndex, isIOSurfaceSupportedFormat);
}
