/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 5, 2024.
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
#include "WebXRReferenceSpace.h"

#if ENABLE(WEBXR)

#include "Document.h"
#include "WebXRFrame.h"
#include "WebXRRigidTransform.h"
#include "WebXRSession.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

static constexpr double DefaultUserHeightInMeters = 1.65;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebXRReferenceSpace);

Ref<WebXRReferenceSpace> WebXRReferenceSpace::create(Document& document, WebXRSession& session, XRReferenceSpaceType type)
{
    // https://immersive-web.github.io/webxr/#xrspace-native-origin
    // The transform from the effective space to the native origin's space is
    // defined by an origin offset, which is an XRRigidTransform initially set
    // to an identity transform.
    return adoptRef(*new WebXRReferenceSpace(document, session, WebXRRigidTransform::create(), type));
}

Ref<WebXRReferenceSpace> WebXRReferenceSpace::create(Document& document, WebXRSession& session, Ref<WebXRRigidTransform>&& offset, XRReferenceSpaceType type)
{
    return adoptRef(*new WebXRReferenceSpace(document, session, WTFMove(offset), type));
}

WebXRReferenceSpace::WebXRReferenceSpace(Document& document, WebXRSession& session, Ref<WebXRRigidTransform>&& offset, XRReferenceSpaceType type)
    : WebXRSpace(document, WTFMove(offset))
    , m_session(session)
    , m_type(type)
{
}

WebXRReferenceSpace::~WebXRReferenceSpace() = default;


std::optional<TransformationMatrix> WebXRReferenceSpace::nativeOrigin() const
{
    if (!m_session)
        return std::nullopt;

    TransformationMatrix identity;

    // We assume that poses got from the devices are in local space.
    // This will require more complex logic if we add ports with different default coordinates.
    switch (m_type) {
    case XRReferenceSpaceType::Viewer: {
        // Return the current pose. Content rendered in viewer pose will stay in fixed point on HMDs.
        auto& data = m_session->frameData();
        return WebXRFrame::matrixFromPose(data.origin);
    }
    case XRReferenceSpaceType::Local:
        // Data from the device is already in local, use the identity matrix.
        return identity;
    case XRReferenceSpaceType::Unbounded:
        // Local and unbounded use the same device space, use the identity matrix.
        return identity;
    case XRReferenceSpaceType::LocalFloor: {
        // Use the floor transform provided by the device or fallback to a default height.
        return floorOriginTransform();
    }
    case XRReferenceSpaceType::BoundedFloor:
    default:
        // BoundedFloor is handled by WebXRBoundedReferenceSpace subclass
        RELEASE_ASSERT_NOT_REACHED();
        return std::nullopt;
    }
}

ExceptionOr<Ref<WebXRReferenceSpace>> WebXRReferenceSpace::getOffsetReferenceSpace(const WebXRRigidTransform& offsetTransform)
{
    if (!m_session)
        return Exception { ExceptionCode::InvalidStateError };

    RefPtr document = downcast<Document>(scriptExecutionContext());
    if (!document)
        return Exception { ExceptionCode::InvalidStateError };

    // https://immersive-web.github.io/webxr/#dom-xrreferencespace-getoffsetreferencespace
    // Set offsetSpaceâ€™s origin offset to the result of multiplying baseâ€™s origin offset by originOffset in the relevant realm of base.
    auto offset = WebXRRigidTransform::create(originOffset().rawTransform() * offsetTransform.rawTransform());

    return create(*document, *m_session.get(), WTFMove(offset), m_type);
}

std::optional<TransformationMatrix> WebXRReferenceSpace::floorOriginTransform() const
{
    if (!m_session)
        return std::nullopt;

    auto& data = m_session->frameData();
    if (!data.floorTransform) {
        TransformationMatrix defautTransform;
        defautTransform.translate3d(0.0, -DefaultUserHeightInMeters, 0.0);
        return defautTransform;
    }

    // https://immersive-web.github.io/webxr/#dom-xrreferencespacetype-local-floor
    // Get floor estimation from the device
    // FIXME: Round to nearest 1cm to prevent fingerprinting
    return WebXRFrame::matrixFromPose(*data.floorTransform);
}

} // namespace WebCore

#endif // ENABLE(WEBXR)
