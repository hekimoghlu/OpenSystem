/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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
#include "WebXRSpace.h"

#if ENABLE(WEBXR)

#include "DOMPointReadOnly.h"
#include "Document.h"
#include "WebXRRigidTransform.h"
#include "WebXRSession.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebXRSpace);

WebXRSpace::WebXRSpace(Document& document, Ref<WebXRRigidTransform>&& offset)
    : ContextDestructionObserver(&document)
    , m_originOffset(WTFMove(offset))
{
}

WebXRSpace::~WebXRSpace() = default;

std::optional<TransformationMatrix> WebXRSpace::effectiveOrigin() const
{
    // https://immersive-web.github.io/webxr/#xrspace-effective-origin
    // The effective origin can be obtained by multiplying origin offset and the native origin.
    auto origin = nativeOrigin();
    if (!origin)
        return std::nullopt;
    return origin.value() * m_originOffset->rawTransform();
}


std::optional<bool> WebXRSpace::isPositionEmulated() const
{
    WebXRSession* xrSession = session();
    if (!xrSession)
        return std::nullopt;

    return xrSession->isPositionEmulated();
}

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebXRViewerSpace);

WebXRViewerSpace::WebXRViewerSpace(Document& document, WebXRSession& session)
    : WebXRSpace(document, WebXRRigidTransform::create())
    , m_session(session)
{
}

WebXRViewerSpace::~WebXRViewerSpace() = default;

std::optional<TransformationMatrix> WebXRViewerSpace::nativeOrigin() const
{
    if (!m_session)
        return std::nullopt;
    return WebXRFrame::matrixFromPose(m_session->frameData().origin);
}


} // namespace WebCore

#endif // ENABLE(WEBXR)
