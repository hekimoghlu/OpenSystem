/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 21, 2025.
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

#import "Device.h"
#import <wtf/Ref.h>
#import <wtf/RefCountedAndCanMakeWeakPtr.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakHashSet.h>
#import <wtf/WeakPtr.h>

using CVPixelBufferRef = struct __CVBuffer*;

struct WGPUExternalTextureImpl {
};

namespace WebGPU {

class CommandEncoder;

class ExternalTexture : public RefCountedAndCanMakeWeakPtr<ExternalTexture>, public WGPUExternalTextureImpl {
    WTF_MAKE_TZONE_ALLOCATED(ExternalTexture);
public:
    static Ref<ExternalTexture> create(CVPixelBufferRef pixelBuffer, WGPUColorSpace colorSpace, Device& device)
    {
        return adoptRef(*new ExternalTexture(pixelBuffer, colorSpace, device));
    }
    static Ref<ExternalTexture> createInvalid(Device& device)
    {
        return adoptRef(*new ExternalTexture(device));
    }

    ~ExternalTexture();

    CVPixelBufferRef pixelBuffer() const { return m_pixelBuffer.get(); }
    WGPUColorSpace colorSpace() const { return m_colorSpace; }

    void destroy();
    void undestroy();
    void setCommandEncoder(CommandEncoder&) const;
    bool isDestroyed() const;

    bool isValid() const;
    void update(CVPixelBufferRef);
    size_t openCommandEncoderCount() const;
    void updateExternalTextures(id<MTLTexture>, id<MTLTexture>);

private:
    ExternalTexture(CVPixelBufferRef, WGPUColorSpace, Device&);
    ExternalTexture(Device&);

    Ref<Device> protectedDevice() const { return m_device; }

    RetainPtr<CVPixelBufferRef> m_pixelBuffer;
    WGPUColorSpace m_colorSpace;
    const Ref<Device> m_device;
    bool m_destroyed { false };
    id<MTLTexture> m_texture0 { nil };
    id<MTLTexture> m_texture1 { nil };
    mutable WeakHashSet<CommandEncoder> m_commandEncoders;
};

} // namespace WebGPU

