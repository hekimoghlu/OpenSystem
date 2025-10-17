/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 11, 2024.
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
#include "WebXRBoundedReferenceSpace.h"

#if ENABLE(WEBXR)

#include "DOMPointReadOnly.h"
#include "Document.h"
#include "WebXRRigidTransform.h"
#include "WebXRSession.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

// https://immersive-web.github.io/webxr/#xrboundedreferencespace-native-bounds-geometry
// It is suggested that points of the native bounds geometry be quantized to the nearest 5cm.
static constexpr float BoundsPrecisionInMeters = 0.05; 
// A valid polygon has at least 3 vertices.
static constexpr int MinimumBoundsVertices = 3; 

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebXRBoundedReferenceSpace);

Ref<WebXRBoundedReferenceSpace> WebXRBoundedReferenceSpace::create(Document& document, WebXRSession& session, XRReferenceSpaceType type)
{
    return adoptRef(*new WebXRBoundedReferenceSpace(document, session, WebXRRigidTransform::create(), type));
}

Ref<WebXRBoundedReferenceSpace> WebXRBoundedReferenceSpace::create(Document& document, WebXRSession& session, Ref<WebXRRigidTransform>&& offset, XRReferenceSpaceType type)
{
    return adoptRef(*new WebXRBoundedReferenceSpace(document, session, WTFMove(offset), type));
}

WebXRBoundedReferenceSpace::WebXRBoundedReferenceSpace(Document& document, WebXRSession& session, Ref<WebXRRigidTransform>&& offset, XRReferenceSpaceType type)
    : WebXRReferenceSpace(document, session, WTFMove(offset), type)
{
}

WebXRBoundedReferenceSpace::~WebXRBoundedReferenceSpace() = default;

std::optional<TransformationMatrix> WebXRBoundedReferenceSpace::nativeOrigin() const
{
    // https://immersive-web.github.io/webxr/#dom-xrreferencespacetype-bounded-floor.
    // Bounded floor space should be at the same height as local floor space.
    return floorOriginTransform();
}

const Vector<Ref<DOMPointReadOnly>>& WebXRBoundedReferenceSpace::boundsGeometry()
{
    updateIfNeeded();
    return m_boundsGeometry;
}

ExceptionOr<Ref<WebXRReferenceSpace>> WebXRBoundedReferenceSpace::getOffsetReferenceSpace(const WebXRRigidTransform& offsetTransform)
{
    if (!m_session)
        return Exception { ExceptionCode::InvalidStateError };

    RefPtr document = downcast<Document>(scriptExecutionContext());
    if (!document)
        return Exception { ExceptionCode::InvalidStateError };

    // https://immersive-web.github.io/webxr/#dom-xrreferencespace-getoffsetreferencespace
    // Set offsetSpaceâ€™s origin offset to the result of multiplying baseâ€™s origin offset by originOffset in the relevant realm of base.
    auto offset = WebXRRigidTransform::create(originOffset().rawTransform() * offsetTransform.rawTransform());

    return { create(*document, *m_session.get(), WTFMove(offset), m_type) };
}

// https://immersive-web.github.io/webxr/#dom-xrboundedreferencespace-boundsgeometry
void WebXRBoundedReferenceSpace::updateIfNeeded()
{
    if (!m_session)
        return;

    auto& frameData = m_session->frameData();
    if (frameData.stageParameters.id == m_lastUpdateId)
        return;
    m_lastUpdateId = frameData.stageParameters.id;

    m_boundsGeometry.clear();

    if (frameData.stageParameters.bounds.size() >= MinimumBoundsVertices) {
        // Each point has to multiplied by the inverse of originOffset.
        auto transform = valueOrDefault(originOffset().rawTransform().inverse());
        for (auto& point : frameData.stageParameters.bounds) {
            auto mappedPoint = transform.mapPoint(FloatPoint3D(point.x(), 0.0, point.y()));
            m_boundsGeometry.append(DOMPointReadOnly::create(quantize(mappedPoint.x()), quantize(mappedPoint.y()), quantize(mappedPoint.z()), 1.0));
        }
    }
}

// https://immersive-web.github.io/webxr/#quantization
float WebXRBoundedReferenceSpace::quantize(float value)
{
    // Each point in the native bounds geometry MUST also be quantized sufficiently to prevent fingerprinting.
    // For userâ€™s safety, quantized points values MUST NOT fall outside the bounds reported by the platform.
    return std::floor(value / BoundsPrecisionInMeters) * BoundsPrecisionInMeters;
}

} // namespace WebCore

#endif // ENABLE(WEBXR)
