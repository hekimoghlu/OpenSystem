/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 6, 2025.
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

#if ENABLE(WEBXR_LAYERS)

#include "WebXRLayer.h"
#include "XRLayerLayout.h"
#include "XRLayerQuality.h"

#include <wtf/TZoneMalloc.h>

namespace WebCore {

class XRCompositionLayer : public WebXRLayer {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(XRCompositionLayer);
public:
    virtual ~XRCompositionLayer();

    XRLayerLayout layout() const { RELEASE_ASSERT_NOT_REACHED(); }

    bool blendTextureSourceAlpha() const { RELEASE_ASSERT_NOT_REACHED(); }
    [[noreturn]] void setBlendTextureSourceAlpha(bool) { RELEASE_ASSERT_NOT_REACHED(); }

    bool forceMonoPresentation() const { RELEASE_ASSERT_NOT_REACHED(); }
    [[noreturn]] void setForceMonoPresentation(bool) { RELEASE_ASSERT_NOT_REACHED(); }

    float opacity() const { RELEASE_ASSERT_NOT_REACHED(); }
    [[noreturn]] void setOpacity(float) { RELEASE_ASSERT_NOT_REACHED(); }

    uint32_t mipLevels() const { RELEASE_ASSERT_NOT_REACHED(); }

    XRLayerQuality quality() const { RELEASE_ASSERT_NOT_REACHED(); }
    [[noreturn]] void setQuality(XRLayerQuality) { RELEASE_ASSERT_NOT_REACHED(); }

    bool needsRedraw() const { RELEASE_ASSERT_NOT_REACHED(); }

    [[noreturn]] void destroy() { RELEASE_ASSERT_NOT_REACHED(); }
protected:
    explicit XRCompositionLayer(ScriptExecutionContext*);
};

} // namespace WebCore

#endif // ENABLE(WEBXR_LAYERS)
