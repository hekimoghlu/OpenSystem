/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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
#include "WebXRRenderState.h"

#if ENABLE(WEBXR)

#include "XRRenderStateInit.h"
#include <wtf/MathExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebXRRenderState);

Ref<WebXRRenderState> WebXRRenderState::create(XRSessionMode mode)
{
    // https://immersive-web.github.io/webxr/#initialize-the-render-state
    // depthNear, depthFar and baseLayer are initialized in the class definition
    return adoptRef(*new WebXRRenderState(mode == XRSessionMode::Inline ? std::make_optional(piOverTwoDouble) : std::nullopt));
}

WebXRRenderState::WebXRRenderState(std::optional<double> inlineVerticalFieldOfView)
    : m_inlineVerticalFieldOfView(inlineVerticalFieldOfView)
{
}

WebXRRenderState::~WebXRRenderState() = default;

Ref<WebXRRenderState> WebXRRenderState::clone() const
{
    return adoptRef(*new WebXRRenderState(*this));
}

WebXRRenderState::WebXRRenderState(const WebXRRenderState& other)
    : m_depth(other.m_depth)
    , m_inlineVerticalFieldOfView(other.m_inlineVerticalFieldOfView)
    , m_baseLayer(other.baseLayer())
{
}

#if ENABLE(WEBXR_LAYERS)
void WebXRRenderState::setLayers(const Vector<Ref<WebXRLayer>>& layers)
{
    m_layers = layers;
}
#endif

} // namespace WebCore

#endif // ENABLE(WEBXR)
