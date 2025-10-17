/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 24, 2023.
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

#include "WebGPURequestAdapterOptions.h"
#include <optional>
#include <wtf/AbstractRefCounted.h>
#include <wtf/CompletionHandler.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace WebCore {
class NativeImage;
class IntSize;
class GraphicsContext;
}

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
class GPUImpl;
class GraphicsContext;
class NativeImage;
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

struct PresentationContextDescriptor;

class GPU : public AbstractRefCounted {
public:
    virtual ~GPU() = default;

    virtual void requestAdapter(const RequestAdapterOptions&, CompletionHandler<void(RefPtr<Adapter>&&)>&&) = 0;

    virtual RefPtr<PresentationContext> createPresentationContext(const PresentationContextDescriptor&) = 0;

    virtual RefPtr<CompositorIntegration> createCompositorIntegration() = 0;
    virtual void paintToCanvas(WebCore::NativeImage&, const WebCore::IntSize&, WebCore::GraphicsContext&) = 0;
    virtual bool isValid(const CompositorIntegration&) const = 0;
    virtual bool isValid(const Buffer&) const = 0;
    virtual bool isValid(const Adapter&) const = 0;
    virtual bool isValid(const BindGroup&) const = 0;
    virtual bool isValid(const BindGroupLayout&) const = 0;
    virtual bool isValid(const CommandBuffer&) const = 0;
    virtual bool isValid(const CommandEncoder&) const = 0;
    virtual bool isValid(const ComputePassEncoder&) const = 0;
    virtual bool isValid(const ComputePipeline&) const = 0;
    virtual bool isValid(const Device&) const = 0;
    virtual bool isValid(const ExternalTexture&) const = 0;
    virtual bool isValid(const PipelineLayout&) const = 0;
    virtual bool isValid(const PresentationContext&) const = 0;
    virtual bool isValid(const QuerySet&) const = 0;
    virtual bool isValid(const Queue&) const = 0;
    virtual bool isValid(const RenderBundleEncoder&) const = 0;
    virtual bool isValid(const RenderBundle&) const = 0;
    virtual bool isValid(const RenderPassEncoder&) const = 0;
    virtual bool isValid(const RenderPipeline&) const = 0;
    virtual bool isValid(const Sampler&) const = 0;
    virtual bool isValid(const ShaderModule&) const = 0;
    virtual bool isValid(const Texture&) const = 0;
    virtual bool isValid(const TextureView&) const = 0;
    virtual bool isValid(const XRBinding&) const = 0;
    virtual bool isValid(const XRSubImage&) const = 0;
    virtual bool isValid(const XRProjectionLayer&) const = 0;
    virtual bool isValid(const XRView&) const = 0;

protected:
    GPU() = default;

private:
    GPU(const GPU&) = delete;
    GPU(GPU&&) = delete;
    GPU& operator=(const GPU&) = delete;
    GPU& operator=(GPU&&) = delete;
};

} // namespace WebCore::WebGPU
