/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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

#include "WebGPUComputePipeline.h"
#include "WebGPUDeviceLostInfo.h"
#include "WebGPUError.h"
#include "WebGPUErrorFilter.h"
#include "WebGPURenderPipeline.h"
#include "WebGPUSupportedFeatures.h"
#include "WebGPUSupportedLimits.h"
#include <optional>
#include <wtf/CompletionHandler.h>
#include <wtf/HashSet.h>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

#if HAVE(IOSURFACE)
#include <IOSurface/IOSurfaceRef.h>
#endif

#if PLATFORM(COCOA) && ENABLE(VIDEO)
#include <WebCore/MediaPlayerIdentifier.h>
#endif

namespace WebCore::WebGPU {

class BindGroup;
struct BindGroupDescriptor;
class BindGroupLayout;
struct BindGroupLayoutDescriptor;
class Buffer;
struct BufferDescriptor;
class CommandBuffer;
class CommandEncoder;
struct CommandEncoderDescriptor;
class ComputePassEncoder;
class ComputePipeline;
struct ComputePipelineDescriptor;
class ExternalTexture;
struct ExternalTextureDescriptor;
class RenderPipeline;
struct RenderPipelineDescriptor;
class PipelineLayout;
struct PipelineLayoutDescriptor;
class PresentationContext;
class QuerySet;
struct QuerySetDescriptor;
class Queue;
class RenderBundleEncoder;
struct RenderBundleEncoderDescriptor;
class RenderPassEncoder;
class RenderPipeline;
struct RenderPipelineDescriptor;
class Sampler;
struct SamplerDescriptor;
class ShaderModule;
struct ShaderModuleDescriptor;
class Surface;
class Texture;
struct TextureDescriptor;
class XRBinding;

class Device : public RefCountedAndCanMakeWeakPtr<Device> {
public:
    virtual ~Device() = default;

    String label() const { return m_label; }

    void setLabel(String&& label)
    {
        m_label = WTFMove(label);
        setLabelInternal(m_label);
    }

    SupportedFeatures& features() { return m_features; }
    const SupportedFeatures& features() const { return m_features; }
    SupportedLimits& limits() { return m_limits; }
    const SupportedLimits& limits() const { return m_limits; }

    virtual Ref<Queue> queue() = 0;

    virtual void destroy() = 0;

    virtual RefPtr<XRBinding> createXRBinding() = 0;
    virtual RefPtr<Buffer> createBuffer(const BufferDescriptor&) = 0;
    virtual RefPtr<Texture> createTexture(const TextureDescriptor&) = 0;
    virtual RefPtr<Sampler> createSampler(const SamplerDescriptor&) = 0;
    virtual RefPtr<ExternalTexture> importExternalTexture(const ExternalTextureDescriptor&) = 0;
#if PLATFORM(COCOA) && ENABLE(VIDEO)
    virtual void updateExternalTexture(const WebCore::WebGPU::ExternalTexture&, const WebCore::MediaPlayerIdentifier&) = 0;
#endif

    virtual RefPtr<BindGroupLayout> createBindGroupLayout(const BindGroupLayoutDescriptor&) = 0;
    virtual RefPtr<PipelineLayout> createPipelineLayout(const PipelineLayoutDescriptor&) = 0;
    virtual RefPtr<BindGroup> createBindGroup(const BindGroupDescriptor&) = 0;

    virtual RefPtr<ShaderModule> createShaderModule(const ShaderModuleDescriptor&) = 0;
    virtual RefPtr<ComputePipeline> createComputePipeline(const ComputePipelineDescriptor&) = 0;
    virtual RefPtr<RenderPipeline> createRenderPipeline(const RenderPipelineDescriptor&) = 0;
    virtual void createComputePipelineAsync(const ComputePipelineDescriptor&, CompletionHandler<void(RefPtr<ComputePipeline>&&, String&&)>&&) = 0;
    virtual void createRenderPipelineAsync(const RenderPipelineDescriptor&, CompletionHandler<void(RefPtr<RenderPipeline>&&, String&&)>&&) = 0;

    virtual RefPtr<CommandEncoder> createCommandEncoder(const std::optional<CommandEncoderDescriptor>&) = 0;
    virtual RefPtr<RenderBundleEncoder> createRenderBundleEncoder(const RenderBundleEncoderDescriptor&) = 0;

    virtual RefPtr<QuerySet> createQuerySet(const QuerySetDescriptor&) = 0;

    virtual void pushErrorScope(ErrorFilter) = 0;
    virtual void popErrorScope(CompletionHandler<void(bool, std::optional<Error>&&)>&&) = 0;
    virtual void resolveUncapturedErrorEvent(CompletionHandler<void(bool, std::optional<Error>&&)>&&) = 0;
    virtual void resolveDeviceLostPromise(CompletionHandler<void(WebCore::WebGPU::DeviceLostReason)>&&) = 0;
    virtual Ref<CommandEncoder> invalidCommandEncoder() = 0;
    virtual Ref<CommandBuffer> invalidCommandBuffer() = 0;
    virtual Ref<RenderPassEncoder> invalidRenderPassEncoder() = 0;
    virtual Ref<ComputePassEncoder> invalidComputePassEncoder() = 0;
    virtual void pauseAllErrorReporting(bool pause) = 0;

protected:
    Device(Ref<SupportedFeatures>&& features, Ref<SupportedLimits>&& limits)
        : m_features(WTFMove(features))
        , m_limits(WTFMove(limits))
    {
    }

private:
    Device(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(const Device&) = delete;
    Device& operator=(Device&&) = delete;

    virtual void setLabelInternal(const String&) = 0;

    String m_label;
    Ref<SupportedFeatures> m_features;
    Ref<SupportedLimits> m_limits;
};

} // namespace WebCore::WebGPU
