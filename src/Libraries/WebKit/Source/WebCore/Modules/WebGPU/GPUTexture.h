/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 16, 2023.
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

#include "ExceptionOr.h"
#include "GPUIntegralTypes.h"
#include "GPUTextureAspect.h"
#include "GPUTextureDimension.h"
#include "GPUTextureFormat.h"
#include "WebGPUTexture.h"
#include <optional>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GPUDevice;
class GPUTextureView;

struct GPUTextureDescriptor;
struct GPUTextureViewDescriptor;

class GPUTexture : public RefCountedAndCanMakeWeakPtr<GPUTexture> {
public:
    static Ref<GPUTexture> create(Ref<WebGPU::Texture>&& backing, const GPUTextureDescriptor& descriptor, const GPUDevice& device)
    {
        return adoptRef(*new GPUTexture(WTFMove(backing), descriptor, device));
    }

    String label() const;
    void setLabel(String&&);

    ExceptionOr<Ref<GPUTextureView>> createView(const std::optional<GPUTextureViewDescriptor>&) const;

    void destroy();
    bool isDestroyed() const;

    WebGPU::Texture& backing() { return m_backing; }
    const WebGPU::Texture& backing() const { return m_backing; }
    GPUTextureFormat format() const { return m_format; }

    GPUIntegerCoordinateOut width() const;
    GPUIntegerCoordinateOut height() const;
    GPUIntegerCoordinateOut depthOrArrayLayers() const;
    GPUIntegerCoordinateOut mipLevelCount() const;
    GPUSize32Out sampleCount() const;
    GPUTextureDimension dimension() const;
    GPUFlagsConstant usage() const;

    static GPUTextureFormat aspectSpecificFormat(GPUTextureFormat, GPUTextureAspect);
    static uint32_t texelBlockSize(GPUTextureFormat);
    static uint32_t texelBlockWidth(GPUTextureFormat);
    static uint32_t texelBlockHeight(GPUTextureFormat);

    virtual ~GPUTexture();
private:
    GPUTexture(Ref<WebGPU::Texture>&&, const GPUTextureDescriptor&, const GPUDevice&);

    GPUTexture(const GPUTexture&) = delete;
    GPUTexture(GPUTexture&&) = delete;
    GPUTexture& operator=(const GPUTexture&) = delete;
    GPUTexture& operator=(GPUTexture&&) = delete;

    Ref<WebGPU::Texture> m_backing;
    const GPUTextureFormat m_format;
    const GPUIntegerCoordinateOut m_width;
    const GPUIntegerCoordinateOut m_height;
    const GPUIntegerCoordinateOut m_depthOrArrayLayers;
    const GPUIntegerCoordinateOut m_mipLevelCount;
    const GPUSize32Out m_sampleCount;
    const GPUTextureDimension m_dimension;
    const GPUFlagsConstant m_usage;
    Ref<const GPUDevice> m_device;
    bool m_isDestroyed { false };
};

}
