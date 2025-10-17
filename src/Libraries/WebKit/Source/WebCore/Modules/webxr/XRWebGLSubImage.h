/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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

#include "XRSubImage.h"

#include <wtf/TZoneMalloc.h>

namespace WebCore {

class WebGLTexture;

// https://immersive-web.github.io/layers/#xrwebglsubimagetype
class XRWebGLSubImage : public XRSubImage {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(XRWebGLSubImage);
public:
    const WebXRViewport& viewport() const final { RELEASE_ASSERT_NOT_REACHED(); }
    Ref<WebGLTexture> colorTexture() const { RELEASE_ASSERT_NOT_REACHED(); }
    RefPtr<WebGLTexture> depthStencilTexture() const { RELEASE_ASSERT_NOT_REACHED(); }
    RefPtr<WebGLTexture> motionVectorTexture() const { RELEASE_ASSERT_NOT_REACHED(); }

    std::optional<uint32_t> imageIndex() const { RELEASE_ASSERT_NOT_REACHED(); }
    uint32_t colorTextureWidth() const { RELEASE_ASSERT_NOT_REACHED(); }
    uint32_t colorTextureHeight() const { RELEASE_ASSERT_NOT_REACHED(); }
    std::optional<uint32_t> depthStencilTextureWidth() const { RELEASE_ASSERT_NOT_REACHED(); }
    std::optional<uint32_t> depthStencilTextureHeight() const { RELEASE_ASSERT_NOT_REACHED(); }
    std::optional<uint32_t> motionVectorTextureWidth() const { RELEASE_ASSERT_NOT_REACHED(); }
    std::optional<uint32_t> motionVectorTextureHeight() const { RELEASE_ASSERT_NOT_REACHED(); }
};

} // namespace WebCore

#endif // ENABLE(WEBXR_LAYERS)
