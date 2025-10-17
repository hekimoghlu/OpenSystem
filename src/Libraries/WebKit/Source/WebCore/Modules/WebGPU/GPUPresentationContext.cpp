/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 4, 2022.
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
#include "GPUPresentationContext.h"

#include "GPUCanvasConfiguration.h"
#include "GPUDevice.h"
#include "GPUTexture.h"
#include "GPUTextureDescriptor.h"

namespace WebCore {

bool GPUPresentationContext::configure(const GPUCanvasConfiguration& canvasConfiguration, GPUIntegerCoordinate width, GPUIntegerCoordinate height, bool reportValidationErrors)
{
    m_device = canvasConfiguration.device.get();
    m_currentTexture = nullptr;
    m_textureDescriptor = GPUTextureDescriptor {
        { "canvas backing"_s },
        GPUExtent3DDict { width, height, 1 },
        1,
        1,
        GPUTextureDimension::_2d,
        canvasConfiguration.format,
        canvasConfiguration.usage,
        canvasConfiguration.viewFormats
    };

    if (!m_backing->configure(canvasConfiguration.convertToBacking(reportValidationErrors))) {
        ASSERT_NOT_REACHED();
        return false;
    }

    return true;
}

void GPUPresentationContext::unconfigure()
{
    m_currentTexture = nullptr;
    m_backing->unconfigure();
}

RefPtr<GPUTexture> GPUPresentationContext::getCurrentTexture(uint32_t index)
{
    if ((!m_currentTexture || m_currentTexture->isDestroyed()) && m_device.get()) {
        if (auto currentTexture = m_backing->getCurrentTexture(index))
            m_currentTexture = GPUTexture::create(*currentTexture, m_textureDescriptor, *m_device.get()).ptr();
    }

    return m_currentTexture;
}

void GPUPresentationContext::present(uint32_t frameIndex, bool presentBacking)
{
    m_currentTexture = nullptr;
    if (presentBacking)
        m_backing->present(frameIndex, presentBacking);
}

} // namespace WebCore
