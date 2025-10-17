/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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
#include "RemoteAdapterProxy.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteAdapterMessages.h"
#include "RemoteDeviceProxy.h"
#include "WebGPUConvertToBackingContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteAdapterProxy);

RemoteAdapterProxy::RemoteAdapterProxy(String&& name, WebCore::WebGPU::SupportedFeatures& features, WebCore::WebGPU::SupportedLimits& limits, bool isFallbackAdapter, bool xrCompatible, RemoteGPUProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    : Adapter(WTFMove(name), features, limits, isFallbackAdapter)
    , m_backing(identifier)
    , m_convertToBackingContext(convertToBackingContext)
    , m_parent(parent)
    , m_xrCompatible(xrCompatible)
{
}

RemoteAdapterProxy::~RemoteAdapterProxy()
{
    auto sendResult = send(Messages::RemoteAdapter::Destruct());
    UNUSED_VARIABLE(sendResult);
}

void RemoteAdapterProxy::requestDevice(const WebCore::WebGPU::DeviceDescriptor& descriptor, CompletionHandler<void(RefPtr<WebCore::WebGPU::Device>&&)>&& callback)
{
    Ref convertToBackingContext = m_convertToBackingContext;
    auto convertedDescriptor = convertToBackingContext->convertToBacking(descriptor);
    ASSERT(convertedDescriptor);
    if (!convertedDescriptor)
        return callback(nullptr);

    auto identifier = WebGPUIdentifier::generate();
    auto queueIdentifier = WebGPUIdentifier::generate();
    auto sendResult = sendSync(Messages::RemoteAdapter::RequestDevice(*convertedDescriptor, identifier, queueIdentifier));
    if (!sendResult.succeeded())
        return callback(nullptr);

    auto [supportedFeatures, supportedLimits] = sendResult.takeReply();
    if (!supportedLimits.maxTextureDimension2D) {
        callback(nullptr);
        return;
    }

    auto resultSupportedFeatures = WebCore::WebGPU::SupportedFeatures::create(WTFMove(supportedFeatures.features));
    auto resultSupportedLimits = WebCore::WebGPU::SupportedLimits::create(
        supportedLimits.maxTextureDimension1D,
        supportedLimits.maxTextureDimension2D,
        supportedLimits.maxTextureDimension3D,
        supportedLimits.maxTextureArrayLayers,
        supportedLimits.maxBindGroups,
        supportedLimits.maxBindGroupsPlusVertexBuffers,
        supportedLimits.maxBindingsPerBindGroup,
        supportedLimits.maxDynamicUniformBuffersPerPipelineLayout,
        supportedLimits.maxDynamicStorageBuffersPerPipelineLayout,
        supportedLimits.maxSampledTexturesPerShaderStage,
        supportedLimits.maxSamplersPerShaderStage,
        supportedLimits.maxStorageBuffersPerShaderStage,
        supportedLimits.maxStorageTexturesPerShaderStage,
        supportedLimits.maxUniformBuffersPerShaderStage,
        supportedLimits.maxUniformBufferBindingSize,
        supportedLimits.maxStorageBufferBindingSize,
        supportedLimits.minUniformBufferOffsetAlignment,
        supportedLimits.minStorageBufferOffsetAlignment,
        supportedLimits.maxVertexBuffers,
        supportedLimits.maxBufferSize,
        supportedLimits.maxVertexAttributes,
        supportedLimits.maxVertexBufferArrayStride,
        supportedLimits.maxInterStageShaderComponents,
        supportedLimits.maxInterStageShaderVariables,
        supportedLimits.maxColorAttachments,
        supportedLimits.maxColorAttachmentBytesPerSample,
        supportedLimits.maxComputeWorkgroupStorageSize,
        supportedLimits.maxComputeInvocationsPerWorkgroup,
        supportedLimits.maxComputeWorkgroupSizeX,
        supportedLimits.maxComputeWorkgroupSizeY,
        supportedLimits.maxComputeWorkgroupSizeZ,
        supportedLimits.maxComputeWorkgroupsPerDimension,
        supportedLimits.maxStorageBuffersInFragmentStage,
        supportedLimits.maxStorageTexturesInFragmentStage
    );
    auto result = RemoteDeviceProxy::create(WTFMove(resultSupportedFeatures), WTFMove(resultSupportedLimits), *this, convertToBackingContext, identifier, queueIdentifier);
    result->setLabel(WTFMove(convertedDescriptor->label));
    callback(WTFMove(result));
}

bool RemoteAdapterProxy::xrCompatible()
{
    return m_xrCompatible;
}

} // namespace WebKit::WebGPU

#endif // HAVE(GPU_PROCESS)
