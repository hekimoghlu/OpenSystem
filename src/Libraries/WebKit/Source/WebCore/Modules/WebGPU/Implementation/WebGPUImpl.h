/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 11, 2022.
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

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPU.h"
#include "WebGPUConvertToBackingContext.h"
#include "WebGPUPtr.h"
#include <WebGPU/WebGPU.h>
#include <WebGPU/WebGPUExt.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Deque.h>
#include <wtf/Function.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class GraphicsContext;
class IntSize;
class NativeImage;
}

namespace WebCore::WebGPU {

class Adapter;
class Buffer;
class BindGroup;
class BindGroupLayout;
class CompositorIntegration;
class CommandBuffer;
class CommandEncoder;
class ComputePassEncoder;
class ComputePipeline;
class ConvertToBackingContext;
class Device;
class ExternalTexture;
class PipelineLayout;
class PresentationContext;
class QuerySet;
class Queue;
class RenderBundleEncoder;
class RenderBundle;
class RenderPassEncoder;
class RenderPipeline;
class Sampler;
class ShaderModule;
class Texture;
class TextureView;
class XRBinding;
class XRProjectionLayer;
class XRSubImage;

class GPUImpl final : public GPU, public RefCounted<GPUImpl> {
    WTF_MAKE_TZONE_ALLOCATED(GPUImpl);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<GPUImpl> create(WebGPUPtr<WGPUInstance>&& instance, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new GPUImpl(WTFMove(instance), convertToBackingContext));
    }

    virtual ~GPUImpl();

    void paintToCanvas(WebCore::NativeImage&, const WebCore::IntSize&, WebCore::GraphicsContext&) final;

private:
    friend class DowncastConvertToBackingContext;

    GPUImpl(WebGPUPtr<WGPUInstance>&&, ConvertToBackingContext&);

    GPUImpl(const GPUImpl&) = delete;
    GPUImpl(GPUImpl&&) = delete;
    GPUImpl& operator=(const GPUImpl&) = delete;
    GPUImpl& operator=(GPUImpl&&) = delete;

    WGPUInstance backing() const { return m_backing.get(); }

    void requestAdapter(const RequestAdapterOptions&, CompletionHandler<void(RefPtr<Adapter>&&)>&&) final;

    RefPtr<PresentationContext> createPresentationContext(const PresentationContextDescriptor&) final;

    RefPtr<CompositorIntegration> createCompositorIntegration() final;
    bool isValid(const CompositorIntegration&) const final;
    bool isValid(const Buffer&) const final;
    bool isValid(const Adapter&) const final;
    bool isValid(const BindGroup&) const final;
    bool isValid(const BindGroupLayout&) const final;
    bool isValid(const CommandBuffer&) const final;
    bool isValid(const CommandEncoder&) const final;
    bool isValid(const ComputePassEncoder&) const final;
    bool isValid(const ComputePipeline&) const final;
    bool isValid(const Device&) const final;
    bool isValid(const ExternalTexture&) const final;
    bool isValid(const PipelineLayout&) const final;
    bool isValid(const PresentationContext&) const final;
    bool isValid(const QuerySet&) const final;
    bool isValid(const Queue&) const final;
    bool isValid(const RenderBundleEncoder&) const final;
    bool isValid(const RenderBundle&) const final;
    bool isValid(const RenderPassEncoder&) const final;
    bool isValid(const RenderPipeline&) const final;
    bool isValid(const Sampler&) const final;
    bool isValid(const ShaderModule&) const final;
    bool isValid(const Texture&) const final;
    bool isValid(const TextureView&) const final;
    bool isValid(const XRBinding&) const final;
    bool isValid(const XRSubImage&) const final;
    bool isValid(const XRProjectionLayer&) const final;
    bool isValid(const XRView&) const final;

    WebGPUPtr<WGPUInstance> m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
