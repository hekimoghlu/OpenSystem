/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 11, 2024.
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
#include "WebXRView.h"

#if ENABLE(WEBXR)

#include "WebXRFrame.h"
#include "WebXRRigidTransform.h"
#include <JavaScriptCore/TypedArrayInlines.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

// Arbitrary value for minimum viewport scaling.
// Below this threshold the resulting viewport would be too pixelated.
static constexpr double kMinViewportScale = 0.1;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebXRView);

Ref<WebXRView> WebXRView::create(Ref<WebXRFrame>&& frame, XREye eye, Ref<WebXRRigidTransform>&& transform, Ref<Float32Array>&& projection)
{
    return adoptRef(*new WebXRView(WTFMove(frame), eye, WTFMove(transform), WTFMove(projection)));
}

WebXRView::WebXRView(Ref<WebXRFrame>&& frame, XREye eye, Ref<WebXRRigidTransform>&& transform, Ref<Float32Array>&& projection)
    : m_frame(WTFMove(frame))
    , m_eye(eye)
    , m_transform(WTFMove(transform))
    , m_projection(WTFMove(projection))
{
}

WebXRView::~WebXRView() = default;

// https://immersive-web.github.io/webxr/#dom-xrview-recommendedviewportscale
std::optional<double> WebXRView::recommendedViewportScale() const
{
    // Return null if the system does not implement a heuristic or method for determining a recommended scale.
    return std::nullopt;
}

// https://immersive-web.github.io/webxr/#dom-xrview-requestviewportscale
void WebXRView::requestViewportScale(std::optional<double> value)
{
    if (!value || *value <= 0.0)
        return;
    m_requestedViewportScale = std::clamp(*value, kMinViewportScale, 1.0);
}


} // namespace WebCore

#endif // ENABLE(WEBXR)
