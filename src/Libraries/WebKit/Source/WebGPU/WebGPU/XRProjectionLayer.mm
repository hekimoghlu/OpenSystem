/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 26, 2025.
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
#import "XRProjectionLayer.h"

#import "APIConversions.h"
#import "Device.h"
#import "MetalSPI.h"
#import <wtf/CheckedArithmetic.h>
#import <wtf/StdLibExtras.h>

namespace WebGPU {

XRProjectionLayer::XRProjectionLayer(bool, Device& device)
    : m_sharedEvent(std::make_pair(nil, 0))
    , m_device(device)
{
    m_colorTextures = [NSMutableDictionary dictionary];
    m_depthTextures = [NSMutableDictionary dictionary];
}

XRProjectionLayer::XRProjectionLayer(Device& device)
    : m_sharedEvent(std::make_pair(nil, 0))
    , m_device(device)
{
}

XRProjectionLayer::~XRProjectionLayer() = default;

void XRProjectionLayer::setLabel(String&&)
{
}

bool XRProjectionLayer::isValid() const
{
    return !!m_colorTextures;
}

id<MTLTexture> XRProjectionLayer::colorTexture() const
{
    return m_colorTexture;
}

id<MTLTexture> XRProjectionLayer::depthTexture() const
{
    return m_depthTexture;
}

const std::pair<id<MTLSharedEvent>, uint64_t>& XRProjectionLayer::completionEvent() const
{
    return m_sharedEvent;
}

size_t XRProjectionLayer::reusableTextureIndex() const
{
    return m_reusableTextureIndex;
}

void XRProjectionLayer::startFrame(size_t frameIndex, WTF::MachSendRight&& colorBuffer, WTF::MachSendRight&& depthBuffer, WTF::MachSendRight&& completionSyncEvent, size_t reusableTextureIndex)
{
#if !PLATFORM(IOS_FAMILY_SIMULATOR) && !PLATFORM(WATCHOS)
    id<MTLDevice> device = m_device->device();
    m_reusableTextureIndex = reusableTextureIndex;
    NSNumber* textureKey = @(reusableTextureIndex);
    if (colorBuffer.sendRight()) {
        MTLSharedTextureHandle* m_colorHandle = [[MTLSharedTextureHandle alloc] initWithMachPort:colorBuffer.sendRight()];
        m_colorTexture = [device newSharedTextureWithHandle:m_colorHandle];
        [m_colorTextures setObject:m_colorTexture forKey:textureKey];
    } else
        m_colorTexture = [m_colorTextures objectForKey:textureKey];

    if (depthBuffer.sendRight()) {
        MTLSharedTextureHandle* m_depthHandle = [[MTLSharedTextureHandle alloc] initWithMachPort:depthBuffer.sendRight()];
        m_depthTexture = [device newSharedTextureWithHandle:m_depthHandle];
        [m_depthTextures setObject:m_depthTexture forKey:textureKey];
    } else
        m_depthTexture = [m_depthTextures objectForKey:textureKey];

    if (completionSyncEvent.sendRight())
        m_sharedEvent = std::make_pair([(id<MTLDeviceSPI>)device newSharedEventWithMachPort:completionSyncEvent.sendRight()], frameIndex);
#else
    UNUSED_PARAM(frameIndex);
    UNUSED_PARAM(colorBuffer);
    UNUSED_PARAM(depthBuffer);
    UNUSED_PARAM(completionSyncEvent);
    UNUSED_PARAM(reusableTextureIndex);
#endif
}

Ref<XRProjectionLayer> XRBinding::createXRProjectionLayer(WGPUTextureFormat colorFormat, WGPUTextureFormat* optionalDepthStencilFormat, WGPUTextureUsageFlags flags, double scale)
{
    UNUSED_PARAM(colorFormat);
    UNUSED_PARAM(optionalDepthStencilFormat);
    UNUSED_PARAM(flags);
    UNUSED_PARAM(scale);

    return XRProjectionLayer::create(protectedDevice().get());
}

} // namespace WebGPU

#pragma mark WGPU Stubs

void wgpuXRProjectionLayerReference(WGPUXRProjectionLayer projectionLayer)
{
    WebGPU::fromAPI(projectionLayer).ref();
}

void wgpuXRProjectionLayerRelease(WGPUXRProjectionLayer projectionLayer)
{
    WebGPU::fromAPI(projectionLayer).deref();
}

void wgpuXRProjectionLayerStartFrame(WGPUXRProjectionLayer layer, size_t frameIndex, WTF::MachSendRight&& colorBuffer, WTF::MachSendRight&& depthBuffer, WTF::MachSendRight&& completionSyncEvent, size_t reusableTextureIndex)
{
    WebGPU::protectedFromAPI(layer)->startFrame(frameIndex, WTFMove(colorBuffer), WTFMove(depthBuffer), WTFMove(completionSyncEvent), reusableTextureIndex);
}
