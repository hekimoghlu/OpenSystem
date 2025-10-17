/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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

#include "WebGPUColor.h"
#include "WebGPUComputePassTimestampWrites.h"
#include "WebGPUError.h"
#include "WebGPUExtent3D.h"
#include "WebGPUIdentifier.h"
#include "WebGPUOrigin2D.h"
#include "WebGPUOrigin3D.h"
#include "WebGPURenderPassTimestampWrites.h"
#include <WebCore/WebGPUColor.h>
#include <WebCore/WebGPUComputePassTimestampWrites.h>
#include <WebCore/WebGPUError.h>
#include <WebCore/WebGPUExtent3D.h>
#include <WebCore/WebGPUOrigin2D.h>
#include <WebCore/WebGPUOrigin3D.h>
#include <WebCore/WebGPURenderPassTimestampWrites.h>
#include <wtf/RefCounted.h>

namespace WebCore::WebGPU {

class Adapter;
class BindGroup;
struct BindGroupDescriptor;
struct BindGroupEntry;
class BindGroupLayout;
struct BindGroupLayoutDescriptor;
struct BindGroupLayoutEntry;
struct BlendComponent;
struct BlendState;
class Buffer;
struct BufferBinding;
struct BufferBindingLayout;
struct BufferDescriptor;
struct CanvasConfiguration;
struct ColorTargetState;
class CommandBuffer;
struct CommandBufferDescriptor;
class CommandEncoder;
struct CommandEncoderDescriptor;
class CompilationMessage;
class CompositorIntegration;
struct ComputePassDescriptor;
class ComputePassEncoder;
class ComputePipeline;
struct ComputePipelineDescriptor;
struct DepthStencilState;
class Device;
struct DeviceDescriptor;
class ExternalTexture;
struct ExternalTextureBindingLayout;
struct ExternalTextureDescriptor;
struct FragmentState;
class GPU;
struct Identifier;
struct ImageCopyBuffer;
struct ImageCopyExternalImage;
struct ImageCopyTexture;
struct ImageCopyTextureTagged;
struct ImageDataLayout;
class InternalError;
struct MultisampleState;
struct ObjectDescriptorBase;
class OutOfMemoryError;
struct PipelineDescriptorBase;
class PipelineLayout;
struct PipelineLayoutDescriptor;
struct CanvasConfiguration;
class PresentationContext;
struct PresentationContextDescriptor;
struct PrimitiveState;
struct ProgrammableStage;
class QuerySet;
struct QuerySetDescriptor;
class Queue;
class RenderBundle;
struct RenderBundleDescriptor;
class RenderBundleEncoder;
struct RenderBundleEncoderDescriptor;
struct RenderPassColorAttachment;
struct RenderPassDepthStencilAttachment;
struct RenderPassDescriptor;
class RenderPassEncoder;
struct RenderPassLayout;
class RenderPipeline;
struct RenderPipelineDescriptor;
struct RequestAdapterOptions;
class Sampler;
struct SamplerBindingLayout;
struct SamplerDescriptor;
class ShaderModule;
struct ShaderModuleCompilationHint;
struct ShaderModuleDescriptor;
struct StencilFaceState;
struct StorageTextureBindingLayout;
class SupportedFeatures;
class SupportedLimits;
class Texture;
struct TextureBindingLayout;
struct TextureDescriptor;
class TextureView;
struct TextureViewDescriptor;
class ValidationError;
struct VertexAttribute;
struct VertexBufferLayout;
struct VertexState;
class XRBinding;
class XRProjectionLayer;
class XRSubImage;
class XRView;

} // namespace WebCore::WebGPU

namespace WebKit::WebGPU {

struct BindGroupDescriptor;
struct BindGroupEntry;
struct BindGroupLayoutDescriptor;
struct BindGroupLayoutEntry;
struct BlendComponent;
struct BlendState;
struct BufferBinding;
struct BufferBindingLayout;
struct BufferDescriptor;
struct CanvasConfiguration;
struct ColorTargetState;
struct CommandBufferDescriptor;
struct CommandEncoderDescriptor;
struct CompilationMessage;
struct ComputePassDescriptor;
struct ComputePipelineDescriptor;
struct DepthStencilState;
struct DeviceDescriptor;
struct ExternalTextureBindingLayout;
struct ExternalTextureDescriptor;
struct FragmentState;
struct Identifier;
struct ImageCopyBuffer;
struct ImageCopyExternalImage;
struct ImageCopyTexture;
struct ImageCopyTextureTagged;
struct ImageDataLayout;
struct InternalError;
struct MultisampleState;
struct ObjectDescriptorBase;
struct OutOfMemoryError;
struct PipelineDescriptorBase;
struct PipelineLayoutDescriptor;
struct PresentationContextDescriptor;
struct CanvasConfiguration;
struct PrimitiveState;
struct ProgrammableStage;
struct QuerySetDescriptor;
class RemoteCompositorIntegrationProxy;
struct RenderBundleDescriptor;
struct RenderBundleEncoderDescriptor;
struct RenderPassColorAttachment;
struct RenderPassDepthStencilAttachment;
struct RenderPassDescriptor;
struct RenderPassLayout;
struct RenderPipelineDescriptor;
struct RequestAdapterOptions;
struct SamplerBindingLayout;
struct SamplerDescriptor;
struct ShaderModuleCompilationHint;
struct ShaderModuleDescriptor;
struct StencilFaceState;
struct StorageTextureBindingLayout;
struct SupportedFeatures;
struct SupportedLimits;
struct TextureBindingLayout;
struct TextureDescriptor;
struct TextureViewDescriptor;
struct ValidationError;
struct VertexAttribute;
struct VertexBufferLayout;
struct VertexState;

class ConvertToBackingContext : public RefCounted<ConvertToBackingContext> {
public:
    virtual ~ConvertToBackingContext() = default;

    std::optional<BindGroupDescriptor> convertToBacking(const WebCore::WebGPU::BindGroupDescriptor&);
    std::optional<BindGroupEntry> convertToBacking(const WebCore::WebGPU::BindGroupEntry&);
    std::optional<BindGroupLayoutDescriptor> convertToBacking(const WebCore::WebGPU::BindGroupLayoutDescriptor&);
    std::optional<BindGroupLayoutEntry> convertToBacking(const WebCore::WebGPU::BindGroupLayoutEntry&);
    std::optional<BlendComponent> convertToBacking(const WebCore::WebGPU::BlendComponent&);
    std::optional<BlendState> convertToBacking(const WebCore::WebGPU::BlendState&);
    std::optional<BufferBinding> convertToBacking(const WebCore::WebGPU::BufferBinding&);
    std::optional<BufferBindingLayout> convertToBacking(const WebCore::WebGPU::BufferBindingLayout&);
    std::optional<BufferDescriptor> convertToBacking(const WebCore::WebGPU::BufferDescriptor&);
    std::optional<CanvasConfiguration> convertToBacking(const WebCore::WebGPU::CanvasConfiguration&);
    std::optional<Color> convertToBacking(const WebCore::WebGPU::Color&);
    std::optional<ColorDict> convertToBacking(const WebCore::WebGPU::ColorDict&);
    std::optional<ColorTargetState> convertToBacking(const WebCore::WebGPU::ColorTargetState&);
    std::optional<CommandBufferDescriptor> convertToBacking(const WebCore::WebGPU::CommandBufferDescriptor&);
    std::optional<CommandEncoderDescriptor> convertToBacking(const WebCore::WebGPU::CommandEncoderDescriptor&);
    std::optional<CompilationMessage> convertToBacking(const WebCore::WebGPU::CompilationMessage&);
    std::optional<ComputePassDescriptor> convertToBacking(const WebCore::WebGPU::ComputePassDescriptor&);
    std::optional<ComputePassTimestampWrites> convertToBacking(const WebCore::WebGPU::ComputePassTimestampWrites&);
    std::optional<ComputePipelineDescriptor> convertToBacking(const WebCore::WebGPU::ComputePipelineDescriptor&);
    std::optional<DepthStencilState> convertToBacking(const WebCore::WebGPU::DepthStencilState&);
    std::optional<DeviceDescriptor> convertToBacking(const WebCore::WebGPU::DeviceDescriptor&);
    std::optional<Error> convertToBacking(const WebCore::WebGPU::Error&);
    std::optional<Extent3D> convertToBacking(const WebCore::WebGPU::Extent3D&);
    std::optional<Extent3DDict> convertToBacking(const WebCore::WebGPU::Extent3DDict&);
    std::optional<ExternalTextureBindingLayout> convertToBacking(const WebCore::WebGPU::ExternalTextureBindingLayout&);
    std::optional<ExternalTextureDescriptor> convertToBacking(const WebCore::WebGPU::ExternalTextureDescriptor&);
    std::optional<FragmentState> convertToBacking(const WebCore::WebGPU::FragmentState&);
    std::optional<Identifier> convertToBacking(const WebCore::WebGPU::Identifier&);
    std::optional<ImageCopyBuffer> convertToBacking(const WebCore::WebGPU::ImageCopyBuffer&);
    std::optional<ImageCopyExternalImage> convertToBacking(const WebCore::WebGPU::ImageCopyExternalImage&);
    std::optional<ImageCopyTexture> convertToBacking(const WebCore::WebGPU::ImageCopyTexture&);
    std::optional<ImageCopyTextureTagged> convertToBacking(const WebCore::WebGPU::ImageCopyTextureTagged&);
    std::optional<ImageDataLayout> convertToBacking(const WebCore::WebGPU::ImageDataLayout&);
    std::optional<InternalError> convertToBacking(const WebCore::WebGPU::InternalError&);
    std::optional<MultisampleState> convertToBacking(const WebCore::WebGPU::MultisampleState&);
    std::optional<ObjectDescriptorBase> convertToBacking(const WebCore::WebGPU::ObjectDescriptorBase&);
    std::optional<Origin2D> convertToBacking(const WebCore::WebGPU::Origin2D&);
    std::optional<Origin2DDict> convertToBacking(const WebCore::WebGPU::Origin2DDict&);
    std::optional<Origin3D> convertToBacking(const WebCore::WebGPU::Origin3D&);
    std::optional<Origin3DDict> convertToBacking(const WebCore::WebGPU::Origin3DDict&);
    std::optional<OutOfMemoryError> convertToBacking(const WebCore::WebGPU::OutOfMemoryError&);
    std::optional<PipelineDescriptorBase> convertToBacking(const WebCore::WebGPU::PipelineDescriptorBase&);
    std::optional<PipelineLayoutDescriptor> convertToBacking(const WebCore::WebGPU::PipelineLayoutDescriptor&);
    std::optional<PresentationContextDescriptor> convertToBacking(const WebCore::WebGPU::PresentationContextDescriptor&);
    std::optional<PrimitiveState> convertToBacking(const WebCore::WebGPU::PrimitiveState&);
    std::optional<ProgrammableStage> convertToBacking(const WebCore::WebGPU::ProgrammableStage&);
    std::optional<QuerySetDescriptor> convertToBacking(const WebCore::WebGPU::QuerySetDescriptor&);
    std::optional<RenderBundleDescriptor> convertToBacking(const WebCore::WebGPU::RenderBundleDescriptor&);
    std::optional<RenderBundleEncoderDescriptor> convertToBacking(const WebCore::WebGPU::RenderBundleEncoderDescriptor&);
    std::optional<RenderPassColorAttachment> convertToBacking(const WebCore::WebGPU::RenderPassColorAttachment&);
    std::optional<RenderPassDepthStencilAttachment> convertToBacking(const WebCore::WebGPU::RenderPassDepthStencilAttachment&);
    std::optional<RenderPassDescriptor> convertToBacking(const WebCore::WebGPU::RenderPassDescriptor&);
    std::optional<RenderPassLayout> convertToBacking(const WebCore::WebGPU::RenderPassLayout&);
    std::optional<RenderPassTimestampWrites> convertToBacking(const WebCore::WebGPU::RenderPassTimestampWrites&);
    std::optional<RenderPipelineDescriptor> convertToBacking(const WebCore::WebGPU::RenderPipelineDescriptor&);
    std::optional<RequestAdapterOptions> convertToBacking(const WebCore::WebGPU::RequestAdapterOptions&);
    std::optional<SamplerBindingLayout> convertToBacking(const WebCore::WebGPU::SamplerBindingLayout&);
    std::optional<SamplerDescriptor> convertToBacking(const WebCore::WebGPU::SamplerDescriptor&);
    std::optional<ShaderModuleCompilationHint> convertToBacking(const WebCore::WebGPU::ShaderModuleCompilationHint&);
    std::optional<ShaderModuleDescriptor> convertToBacking(const WebCore::WebGPU::ShaderModuleDescriptor&);
    std::optional<StencilFaceState> convertToBacking(const WebCore::WebGPU::StencilFaceState&);
    std::optional<StorageTextureBindingLayout> convertToBacking(const WebCore::WebGPU::StorageTextureBindingLayout&);
    std::optional<SupportedFeatures> convertToBacking(const WebCore::WebGPU::SupportedFeatures&);
    std::optional<SupportedLimits> convertToBacking(const WebCore::WebGPU::SupportedLimits&);
    std::optional<TextureBindingLayout> convertToBacking(const WebCore::WebGPU::TextureBindingLayout&);
    std::optional<TextureDescriptor> convertToBacking(const WebCore::WebGPU::TextureDescriptor&);
    std::optional<TextureViewDescriptor> convertToBacking(const WebCore::WebGPU::TextureViewDescriptor&);
    std::optional<ValidationError> convertToBacking(const WebCore::WebGPU::ValidationError&);
    std::optional<VertexAttribute> convertToBacking(const WebCore::WebGPU::VertexAttribute&);
    std::optional<VertexBufferLayout> convertToBacking(const WebCore::WebGPU::VertexBufferLayout&);
    std::optional<VertexState> convertToBacking(const WebCore::WebGPU::VertexState&);

    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::Adapter&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::BindGroup&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::BindGroupLayout&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::Buffer&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::CommandBuffer&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::CommandEncoder&) = 0;
    virtual const RemoteCompositorIntegrationProxy& convertToRawBacking(const WebCore::WebGPU::CompositorIntegration&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::CompositorIntegration&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::ComputePassEncoder&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::ComputePipeline&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::Device&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::ExternalTexture&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::GPU&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::PipelineLayout&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::PresentationContext&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::QuerySet&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::Queue&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::RenderBundleEncoder&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::RenderBundle&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::RenderPassEncoder&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::RenderPipeline&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::Sampler&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::ShaderModule&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::Texture&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::TextureView&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::XRBinding&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::XRProjectionLayer&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::XRSubImage&) = 0;
    virtual WebGPUIdentifier convertToBacking(const WebCore::WebGPU::XRView&) = 0;
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
