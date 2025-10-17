/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 8, 2022.
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

#if ENABLE(GPU_PROCESS)

#include "ScopedActiveMessageReceiveQueue.h"
#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUIdentifier.h"
#include <functional>
#include <variant>
#include <wtf/HashMap.h>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {
class Adapter;
class BindGroup;
class BindGroupLayout;
class Buffer;
class CommandBuffer;
class CommandEncoder;
class CompositorIntegration;
class ComputePassEncoder;
class ComputePipeline;
class Device;
class ExternalTexture;
class GPU;
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
class XRView;
}

namespace WebKit {
class RemoteAdapter;
class RemoteBindGroup;
class RemoteBindGroupLayout;
class RemoteBuffer;
class RemoteCommandBuffer;
class RemoteCommandEncoder;
class RemoteCompositorIntegration;
class RemoteComputePassEncoder;
class RemoteComputePipeline;
class RemoteDevice;
class RemoteExternalTexture;
class RemotePipelineLayout;
class RemotePresentationContext;
class RemoteQuerySet;
class RemoteQueue;
class RemoteRenderBundleEncoder;
class RemoteRenderBundle;
class RemoteRenderPassEncoder;
class RemoteRenderPipeline;
class RemoteSampler;
class RemoteShaderModule;
class RemoteTexture;
class RemoteTextureView;
class RemoteXRBinding;
class RemoteXRProjectionLayer;
class RemoteXRSubImage;
class RemoteXRView;
}

namespace WebKit::WebGPU {

class ObjectHeap final : public RefCountedAndCanMakeWeakPtr<ObjectHeap>, public WebGPU::ConvertFromBackingContext {
    WTF_MAKE_TZONE_ALLOCATED(ObjectHeap);
public:
    static Ref<ObjectHeap> create()
    {
        return adoptRef(*new ObjectHeap());
    }

    ~ObjectHeap();

    void addObject(WebGPUIdentifier, RemoteAdapter&);
    void addObject(WebGPUIdentifier, RemoteBindGroup&);
    void addObject(WebGPUIdentifier, RemoteBindGroupLayout&);
    void addObject(WebGPUIdentifier, RemoteBuffer&);
    void addObject(WebGPUIdentifier, RemoteCommandBuffer&);
    void addObject(WebGPUIdentifier, RemoteCommandEncoder&);
    void addObject(WebGPUIdentifier, RemoteCompositorIntegration&);
    void addObject(WebGPUIdentifier, RemoteComputePassEncoder&);
    void addObject(WebGPUIdentifier, RemoteComputePipeline&);
    void addObject(WebGPUIdentifier, RemoteDevice&);
    void addObject(WebGPUIdentifier, RemoteExternalTexture&);
    void addObject(WebGPUIdentifier, RemotePipelineLayout&);
    void addObject(WebGPUIdentifier, RemotePresentationContext&);
    void addObject(WebGPUIdentifier, RemoteQuerySet&);
    void addObject(WebGPUIdentifier, RemoteQueue&);
    void addObject(WebGPUIdentifier, RemoteRenderBundleEncoder&);
    void addObject(WebGPUIdentifier, RemoteRenderBundle&);
    void addObject(WebGPUIdentifier, RemoteRenderPassEncoder&);
    void addObject(WebGPUIdentifier, RemoteRenderPipeline&);
    void addObject(WebGPUIdentifier, RemoteSampler&);
    void addObject(WebGPUIdentifier, RemoteShaderModule&);
    void addObject(WebGPUIdentifier, RemoteTexture&);
    void addObject(WebGPUIdentifier, RemoteTextureView&);
    void addObject(WebGPUIdentifier, RemoteXRBinding&);
    void addObject(WebGPUIdentifier, RemoteXRSubImage&);
    void addObject(WebGPUIdentifier, RemoteXRProjectionLayer&);
    void addObject(WebGPUIdentifier, RemoteXRView&);

    void removeObject(WebGPUIdentifier);

    void clear();

    WeakPtr<WebCore::WebGPU::Adapter> convertAdapterFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::BindGroup> convertBindGroupFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::BindGroupLayout> convertBindGroupLayoutFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::Buffer> convertBufferFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::CommandBuffer> convertCommandBufferFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::CommandEncoder> convertCommandEncoderFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::CompositorIntegration> convertCompositorIntegrationFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::ComputePassEncoder> convertComputePassEncoderFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::ComputePipeline> convertComputePipelineFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::Device> convertDeviceFromBacking(WebGPUIdentifier) final;
    ThreadSafeWeakPtr<WebCore::WebGPU::ExternalTexture> convertExternalTextureFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::PipelineLayout> convertPipelineLayoutFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::PresentationContext> convertPresentationContextFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::QuerySet> convertQuerySetFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::Queue> convertQueueFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::RenderBundleEncoder> convertRenderBundleEncoderFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::RenderBundle> convertRenderBundleFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::RenderPassEncoder> convertRenderPassEncoderFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::RenderPipeline> convertRenderPipelineFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::Sampler> convertSamplerFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::ShaderModule> convertShaderModuleFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::Texture> convertTextureFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::TextureView> convertTextureViewFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::XRBinding> convertXRBindingFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::XRSubImage> convertXRSubImageFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::XRProjectionLayer> convertXRProjectionLayerFromBacking(WebGPUIdentifier) final;
    WeakPtr<WebCore::WebGPU::XRView> createXRViewFromBacking(WebGPUIdentifier) final;

    struct ExistsAndValid {
        bool exists { false };
        bool valid { false };
    };
    ExistsAndValid objectExistsAndValid(const WebCore::WebGPU::GPU&, WebGPUIdentifier) const;
private:
    ObjectHeap();

    using Object = std::variant<
        std::monostate,
        IPC::ScopedActiveMessageReceiveQueue<RemoteAdapter>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteBindGroup>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteBindGroupLayout>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteBuffer>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteCommandBuffer>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteCommandEncoder>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteCompositorIntegration>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteComputePassEncoder>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteComputePipeline>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteDevice>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteExternalTexture>,
        IPC::ScopedActiveMessageReceiveQueue<RemotePipelineLayout>,
        IPC::ScopedActiveMessageReceiveQueue<RemotePresentationContext>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteQuerySet>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteQueue>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteRenderBundleEncoder>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteRenderBundle>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteRenderPassEncoder>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteRenderPipeline>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteSampler>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteShaderModule>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteTexture>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteTextureView>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteXRBinding>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteXRSubImage>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteXRProjectionLayer>,
        IPC::ScopedActiveMessageReceiveQueue<RemoteXRView>
    >;

    HashMap<WebGPUIdentifier, Object> m_objects;
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
