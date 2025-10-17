/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 24, 2022.
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
#include "config.h"
#include "GPUSupportedLimits.h"

namespace WebCore {

uint32_t GPUSupportedLimits::maxTextureDimension1D() const
{
    return m_backing->maxTextureDimension1D();
}

uint32_t GPUSupportedLimits::maxTextureDimension2D() const
{
    return m_backing->maxTextureDimension2D();
}

uint32_t GPUSupportedLimits::maxTextureDimension3D() const
{
    return m_backing->maxTextureDimension3D();
}

uint32_t GPUSupportedLimits::maxTextureArrayLayers() const
{
    return m_backing->maxTextureArrayLayers();
}

uint32_t GPUSupportedLimits::maxBindGroups() const
{
    return m_backing->maxBindGroups();
}

uint32_t GPUSupportedLimits::maxBindGroupsPlusVertexBuffers() const
{
    return m_backing->maxBindGroupsPlusVertexBuffers();
}

uint32_t GPUSupportedLimits::maxBindingsPerBindGroup() const
{
    return m_backing->maxBindingsPerBindGroup();
}

uint32_t GPUSupportedLimits::maxDynamicUniformBuffersPerPipelineLayout() const
{
    return m_backing->maxDynamicUniformBuffersPerPipelineLayout();
}

uint32_t GPUSupportedLimits::maxDynamicStorageBuffersPerPipelineLayout() const
{
    return m_backing->maxDynamicStorageBuffersPerPipelineLayout();
}

uint32_t GPUSupportedLimits::maxSampledTexturesPerShaderStage() const
{
    return m_backing->maxSampledTexturesPerShaderStage();
}

uint32_t GPUSupportedLimits::maxSamplersPerShaderStage() const
{
    return m_backing->maxSamplersPerShaderStage();
}

uint32_t GPUSupportedLimits::maxStorageBuffersPerShaderStage() const
{
    return m_backing->maxStorageBuffersPerShaderStage();
}

uint32_t GPUSupportedLimits::maxStorageTexturesPerShaderStage() const
{
    return m_backing->maxStorageTexturesPerShaderStage();
}

uint32_t GPUSupportedLimits::maxUniformBuffersPerShaderStage() const
{
    return m_backing->maxUniformBuffersPerShaderStage();
}

uint64_t GPUSupportedLimits::maxUniformBufferBindingSize() const
{
    return m_backing->maxUniformBufferBindingSize();
}

uint64_t GPUSupportedLimits::maxStorageBufferBindingSize() const
{
    return m_backing->maxStorageBufferBindingSize();
}

uint32_t GPUSupportedLimits::minUniformBufferOffsetAlignment() const
{
    return m_backing->minUniformBufferOffsetAlignment();
}

uint32_t GPUSupportedLimits::minStorageBufferOffsetAlignment() const
{
    return m_backing->minStorageBufferOffsetAlignment();
}

uint32_t GPUSupportedLimits::maxVertexBuffers() const
{
    return m_backing->maxVertexBuffers();
}

uint64_t GPUSupportedLimits::maxBufferSize() const
{
    return m_backing->maxBufferSize();
}

uint32_t GPUSupportedLimits::maxVertexAttributes() const
{
    return m_backing->maxVertexAttributes();
}

uint32_t GPUSupportedLimits::maxVertexBufferArrayStride() const
{
    return m_backing->maxVertexBufferArrayStride();
}

uint32_t GPUSupportedLimits::maxInterStageShaderComponents() const
{
    return m_backing->maxInterStageShaderComponents();
}

uint32_t GPUSupportedLimits::maxInterStageShaderVariables() const
{
    return m_backing->maxInterStageShaderVariables();
}

uint32_t GPUSupportedLimits::maxColorAttachments() const
{
    return m_backing->maxColorAttachments();
}

uint32_t GPUSupportedLimits::maxColorAttachmentBytesPerSample() const
{
    return m_backing->maxColorAttachmentBytesPerSample();
}

uint32_t GPUSupportedLimits::maxComputeWorkgroupStorageSize() const
{
    return m_backing->maxComputeWorkgroupStorageSize();
}

uint32_t GPUSupportedLimits::maxComputeInvocationsPerWorkgroup() const
{
    return m_backing->maxComputeInvocationsPerWorkgroup();
}

uint32_t GPUSupportedLimits::maxComputeWorkgroupSizeX() const
{
    return m_backing->maxComputeWorkgroupSizeX();
}

uint32_t GPUSupportedLimits::maxComputeWorkgroupSizeY() const
{
    return m_backing->maxComputeWorkgroupSizeY();
}

uint32_t GPUSupportedLimits::maxComputeWorkgroupSizeZ() const
{
    return m_backing->maxComputeWorkgroupSizeZ();
}

uint32_t GPUSupportedLimits::maxComputeWorkgroupsPerDimension() const
{
    return m_backing->maxComputeWorkgroupsPerDimension();
}

uint32_t GPUSupportedLimits::maxStorageBuffersInFragmentStage() const
{
    return m_backing->maxStorageBuffersInFragmentStage();
}

uint32_t GPUSupportedLimits::maxStorageTexturesInFragmentStage() const
{
    return m_backing->maxStorageTexturesInFragmentStage();
}

}
