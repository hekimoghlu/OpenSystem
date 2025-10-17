/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 17, 2023.
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
#import "ExternalTexture.h"

#import "APIConversions.h"
#import "Device.h"
#import "TextureView.h"
#import <wtf/CheckedArithmetic.h>
#import <wtf/MathExtras.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/spi/cocoa/IOSurfaceSPI.h>

namespace WebGPU {

Ref<ExternalTexture> Device::createExternalTexture(const WGPUExternalTextureDescriptor& descriptor)
{
    if (!isValid())
        return ExternalTexture::createInvalid(*this);

    return ExternalTexture::create(descriptor.pixelBuffer, descriptor.colorSpace, *this);
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(ExternalTexture);

ExternalTexture::ExternalTexture(CVPixelBufferRef pixelBuffer, WGPUColorSpace colorSpace, Device& device)
    : m_pixelBuffer(pixelBuffer)
    , m_colorSpace(colorSpace)
    , m_device(device)
{
    update(pixelBuffer);
}

ExternalTexture::ExternalTexture(Device& device)
    : m_device(device)
{
}

bool ExternalTexture::isValid() const
{
    return m_pixelBuffer.get() || m_destroyed;
}

ExternalTexture::~ExternalTexture() = default;

void ExternalTexture::destroy()
{
    m_pixelBuffer = nil;
    m_destroyed = true;
    for (Ref commandEncoder : m_commandEncoders)
        commandEncoder->makeSubmitInvalid();

    m_commandEncoders.clear();
}

void ExternalTexture::undestroy()
{
    m_commandEncoders.clear();
    m_destroyed = false;
}

void ExternalTexture::setCommandEncoder(CommandEncoder& commandEncoder) const
{
    CommandEncoder::trackEncoder(commandEncoder, m_commandEncoders);
    commandEncoder.addTexture(m_texture0);
    commandEncoder.addTexture(m_texture1);
    if (isDestroyed())
        commandEncoder.makeSubmitInvalid();
}

void ExternalTexture::updateExternalTextures(id<MTLTexture> texture0, id<MTLTexture> texture1)
{
    m_texture0 = texture0;
    m_texture1 = texture1;
}

bool ExternalTexture::isDestroyed() const
{
    return m_destroyed;
}

void ExternalTexture::update(CVPixelBufferRef pixelBuffer)
{
#if HAVE(IOSURFACE_SET_OWNERSHIP_IDENTITY) && HAVE(TASK_IDENTITY_TOKEN)
    if (IOSurfaceRef ioSurface = CVPixelBufferGetIOSurface(pixelBuffer)) {
        if (auto optionalWebProcessID = protectedDevice()->webProcessID()) {
            if (auto webProcessID = optionalWebProcessID->sendRight())
                IOSurfaceSetOwnershipIdentity(ioSurface, webProcessID, kIOSurfaceMemoryLedgerTagGraphics, 0);
        }
    }
#endif
    m_pixelBuffer = pixelBuffer;
    m_commandEncoders.clear();
    m_destroyed = false;
}

size_t ExternalTexture::openCommandEncoderCount() const
{
    return m_commandEncoders.computeSize();
}

} // namespace WebGPU

#pragma mark WGPU Stubs

void wgpuExternalTextureReference(WGPUExternalTexture externalTexture)
{
    WebGPU::fromAPI(externalTexture).ref();
}

void wgpuExternalTextureRelease(WGPUExternalTexture externalTexture)
{
    WebGPU::fromAPI(externalTexture).deref();
}

void wgpuExternalTextureDestroy(WGPUExternalTexture externalTexture)
{
    WebGPU::protectedFromAPI(externalTexture)->destroy();
}

void wgpuExternalTextureUndestroy(WGPUExternalTexture externalTexture)
{
    WebGPU::protectedFromAPI(externalTexture)->undestroy();
}

void wgpuExternalTextureUpdate(WGPUExternalTexture externalTexture, CVPixelBufferRef pixelBuffer)
{
    WebGPU::protectedFromAPI(externalTexture)->update(pixelBuffer);
}
