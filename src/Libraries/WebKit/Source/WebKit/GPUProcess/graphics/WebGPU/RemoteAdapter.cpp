/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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
#include "RemoteAdapter.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteAdapterMessages.h"
#include "RemoteDevice.h"
#include "RemoteQueue.h"
#include "StreamServerConnection.h"
#include "WebGPUDeviceDescriptor.h"
#include "WebGPUObjectHeap.h"
#include "WebGPUSupportedFeatures.h"
#include "WebGPUSupportedLimits.h"
#include <WebCore/WebGPUAdapter.h>
#include <WebCore/WebGPUDevice.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteAdapter);

RemoteAdapter::RemoteAdapter(GPUConnectionToWebProcess& gpuConnectionToWebProcess, RemoteGPU& gpu, WebCore::WebGPU::Adapter& adapter, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, WebGPUIdentifier identifier)
    : m_backing(adapter)
    , m_objectHeap(objectHeap)
    , m_streamConnection(WTFMove(streamConnection))
    , m_gpuConnectionToWebProcess(gpuConnectionToWebProcess)
    , m_gpu(gpu)
    , m_identifier(identifier)
{
    Ref { m_streamConnection }->startReceivingMessages(*this, Messages::RemoteAdapter::messageReceiverName(), m_identifier.toUInt64());
}

RemoteAdapter::~RemoteAdapter() = default;

void RemoteAdapter::destruct()
{
    protectedObjectHeap()->removeObject(m_identifier);
}

void RemoteAdapter::stopListeningForIPC()
{
    Ref { m_streamConnection }->stopReceivingMessages(Messages::RemoteAdapter::messageReceiverName(), m_identifier.toUInt64());
}

void RemoteAdapter::requestDevice(const WebGPU::DeviceDescriptor& descriptor, WebGPUIdentifier identifier, WebGPUIdentifier queueIdentifier, CompletionHandler<void(WebGPU::SupportedFeatures&&, WebGPU::SupportedLimits&&)>&& callback)
{
    auto convertedDescriptor = m_objectHeap->convertFromBacking(descriptor);
    ASSERT(convertedDescriptor);
    if (!convertedDescriptor) {
        callback({ { } }, { });
        return;
    }

    Ref { m_backing }->requestDevice(*convertedDescriptor, [callback = WTFMove(callback), objectHeap = protectedObjectHeap(), streamConnection = m_streamConnection.copyRef(), identifier, queueIdentifier, gpuConnectionToWebProcess = m_gpuConnectionToWebProcess.get(), gpu = protectedGPU()] (RefPtr<WebCore::WebGPU::Device>&& devicePtr) mutable {
        if (!devicePtr.get() || !gpuConnectionToWebProcess) {
            callback({ }, { });
            return;
        }

        auto device = devicePtr.releaseNonNull();
        auto remoteDevice = RemoteDevice::create(*gpuConnectionToWebProcess, gpu, device, objectHeap, WTFMove(streamConnection), identifier, queueIdentifier);
        objectHeap->addObject(identifier, remoteDevice);
        objectHeap->addObject(queueIdentifier, remoteDevice->queue());
        Ref features = device->features();
        Ref limits = device->limits();
        callback(WebGPU::SupportedFeatures { features->features() }, WebGPU::SupportedLimits {
            limits->maxTextureDimension1D(),
            limits->maxTextureDimension2D(),
            limits->maxTextureDimension3D(),
            limits->maxTextureArrayLayers(),
            limits->maxBindGroups(),
            limits->maxBindGroupsPlusVertexBuffers(),
            limits->maxBindingsPerBindGroup(),
            limits->maxDynamicUniformBuffersPerPipelineLayout(),
            limits->maxDynamicStorageBuffersPerPipelineLayout(),
            limits->maxSampledTexturesPerShaderStage(),
            limits->maxSamplersPerShaderStage(),
            limits->maxStorageBuffersPerShaderStage(),
            limits->maxStorageTexturesPerShaderStage(),
            limits->maxUniformBuffersPerShaderStage(),
            limits->maxUniformBufferBindingSize(),
            limits->maxStorageBufferBindingSize(),
            limits->minUniformBufferOffsetAlignment(),
            limits->minStorageBufferOffsetAlignment(),
            limits->maxVertexBuffers(),
            limits->maxBufferSize(),
            limits->maxVertexAttributes(),
            limits->maxVertexBufferArrayStride(),
            limits->maxInterStageShaderComponents(),
            limits->maxInterStageShaderVariables(),
            limits->maxColorAttachments(),
            limits->maxColorAttachmentBytesPerSample(),
            limits->maxComputeWorkgroupStorageSize(),
            limits->maxComputeInvocationsPerWorkgroup(),
            limits->maxComputeWorkgroupSizeX(),
            limits->maxComputeWorkgroupSizeY(),
            limits->maxComputeWorkgroupSizeZ(),
            limits->maxComputeWorkgroupsPerDimension(),
            limits->maxStorageBuffersInFragmentStage(),
            limits->maxStorageTexturesInFragmentStage(),
        });
    });
}

} // namespace WebKit

#endif // HAVE(GPU_PROCESS)
