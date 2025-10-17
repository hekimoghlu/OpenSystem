/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 9, 2023.
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

#if ENABLE(WEBXR) && USE(OPENXR)

#include "OpenXRSwapchain.h"
#include "OpenXRUtils.h"
#include "PlatformXR.h"

#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>
#include <wtf/Vector.h>

namespace PlatformXR {

class OpenXRLayer {
    WTF_MAKE_TZONE_ALLOCATED(OpenXRLayer);
    WTF_MAKE_NONCOPYABLE(OpenXRLayer);
public:
    virtual ~OpenXRLayer() = default;

    virtual std::optional<FrameData::LayerData> startFrame() = 0;
    virtual XrCompositionLayerBaseHeader* endFrame(const Device::Layer&, XrSpace, const Vector<XrView>&) = 0;

protected:
    OpenXRLayer() = default;
};

class OpenXRLayerProjection final: public OpenXRLayer  {
    WTF_MAKE_TZONE_ALLOCATED(OpenXRLayerProjection);
    WTF_MAKE_NONCOPYABLE(OpenXRLayerProjection);
public:
    static std::unique_ptr<OpenXRLayerProjection> create(XrInstance, XrSession, uint32_t width, uint32_t height, int64_t format, uint32_t sampleCount);
private:
    OpenXRLayerProjection(UniqueRef<OpenXRSwapchain>&&);

    std::optional<FrameData::LayerData> startFrame() final;
    XrCompositionLayerBaseHeader* endFrame(const Device::Layer&, XrSpace, const Vector<XrView>&) final;

    UniqueRef<OpenXRSwapchain> m_swapchain;
    XrCompositionLayerProjection m_layerProjection;
    Vector<XrCompositionLayerProjectionView> m_projectionViews;
};

} // namespace PlatformXR

#endif // ENABLE(WEBXR) && USE(OPENXR)
