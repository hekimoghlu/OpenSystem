/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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

#import <wtf/FastMalloc.h>
#import <wtf/Ref.h>
#import <wtf/RefCounted.h>
#import <wtf/RetainPtr.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/TypeCasts.h>

struct WGPUSurfaceImpl {
};

struct WGPUSwapChainImpl {
};

namespace WebGPU {

class Adapter;
class Device;
class Instance;
class Texture;
class TextureView;

class PresentationContext : public WGPUSurfaceImpl, public WGPUSwapChainImpl, public RefCounted<PresentationContext> {
    WTF_MAKE_TZONE_ALLOCATED(PresentationContext);
public:
    static Ref<PresentationContext> create(const WGPUSurfaceDescriptor&, const Instance&);
    static Ref<PresentationContext> createInvalid()
    {
        return adoptRef(*new PresentationContext());
    }

    virtual ~PresentationContext();

    WGPUTextureFormat getPreferredFormat(const Adapter&);

    virtual void configure(Device&, const WGPUSwapChainDescriptor&);
    virtual void unconfigure();

    virtual void present(uint32_t);
    virtual Texture* getCurrentTexture(uint32_t);
    virtual TextureView* getCurrentTextureView(); // FIXME: This should return a TextureView&.

    virtual bool isPresentationContextIOSurface() const { return false; }
    virtual bool isPresentationContextCoreAnimation() const { return false; }
    virtual RetainPtr<CGImageRef> getTextureAsNativeImage(uint32_t, bool&) { return nullptr; }

    virtual bool isValid() { return false; }
protected:
    explicit PresentationContext();
};

} // namespace WebGPU

#define SPECIALIZE_TYPE_TRAITS_WEBGPU_PRESENTATION_CONTEXT(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebGPU::ToValueTypeName) \
    static bool isType(const WebGPU::PresentationContext& presentationContext) { return presentationContext.predicate; } \
SPECIALIZE_TYPE_TRAITS_END()
