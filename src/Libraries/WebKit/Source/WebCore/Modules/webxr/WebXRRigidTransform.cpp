/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 22, 2022.
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
#include "WebXRRigidTransform.h"

#if ENABLE(WEBXR)

#include "DOMPointReadOnly.h"
#include "TransformationMatrix.h"
#include <JavaScriptCore/TypedArrayInlines.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

static bool normalizeQuaternion(DOMPointInit& q)
{
    const double length = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (WTF::areEssentiallyEqual<double>(length, 0))
        return false;
    q.x /= length;
    q.y /= length;
    q.z /= length;
    q.w /= length;
    return true;
}

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebXRRigidTransform);

Ref<WebXRRigidTransform> WebXRRigidTransform::create()
{
    return adoptRef(*new WebXRRigidTransform({ }, { }));
}

Ref<WebXRRigidTransform> WebXRRigidTransform::create(const TransformationMatrix& transform)
{
    return adoptRef(*new WebXRRigidTransform(transform));
}


ExceptionOr<Ref<WebXRRigidTransform>> WebXRRigidTransform::create(const DOMPointInit& position, const DOMPointInit& orientation)
{
    // The XRRigidTransform(position, orientation) constructor MUST perform the following steps when invoked:
    //   1. Let transform be a new XRRigidTransform.

    //   2. If position is not a DOMPointInit initialize transformâ€™s position to { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }.
    //   3. If positionâ€™s w value is not 1.0, throw a TypeError.
    //   4. Else initialize transformâ€™s positionâ€™s x value to positionâ€™s x dictionary member, y value to positionâ€™s y dictionary member, z value to positionâ€™s z dictionary member and w to 1.0.
    if (position.w != 1.0)
        return Exception { ExceptionCode::TypeError };
    DOMPointInit positionInit { position.x, position.y, position.z, 1 };

    //   5. If orientation is not a DOMPointInit initialize transformâ€™s orientation to { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }.
    //   6. Else initialize transformâ€™s orientationâ€™s x value to orientationâ€™s x dictionary member, y value to orientationâ€™s y dictionary member, z value to orientationâ€™s z dictionary member and w value to orientationâ€™s w dictionary member.
    //   7. Normalize x, y, z, and w components of transformâ€™s orientation.
    DOMPointInit orientationInit { orientation.x, orientation.y, orientation.z, orientation.w };
    if (!normalizeQuaternion(orientationInit))
        return Exception { ExceptionCode::InvalidStateError };

    //   8. Return transform.
    return adoptRef(*new WebXRRigidTransform(positionInit, orientationInit));
}

WebXRRigidTransform::WebXRRigidTransform(const DOMPointInit& position, const DOMPointInit& orientation)
    : m_position(DOMPointReadOnly::create(position))
    , m_orientation(DOMPointReadOnly::create(orientation))
{
    TransformationMatrix translation;
    translation.translate3d(position.x, position.y, position.z);
    auto rotation = TransformationMatrix::fromQuaternion({ orientation.x, orientation.y, orientation.z, orientation.w });
    m_rawTransform = translation * rotation;
}

WebXRRigidTransform::WebXRRigidTransform(const TransformationMatrix& transform)
    : m_position(DOMPointReadOnly::create({ }))
    , m_orientation(DOMPointReadOnly::create({ }))
    , m_rawTransform(transform)
{
    if (transform.isIdentity()) {
        // TransformationMatrix::decompose returns a empty quaternion instead of unit quaternion for Identity.
        // WebXR tests expect a unit quaternion for this case.
        return;
    }

    TransformationMatrix::Decomposed4Type decomp = { };
    if (!transform.decompose4(decomp))
        return;

    m_position = DOMPointReadOnly::create(decomp.translateX, decomp.translateY, decomp.translateZ, 1.0f);

    DOMPointInit orientationInit { decomp.quaternion.x, decomp.quaternion.y, decomp.quaternion.z, decomp.quaternion.w };
    normalizeQuaternion(orientationInit);
    m_orientation = DOMPointReadOnly::create(orientationInit);
}

WebXRRigidTransform::~WebXRRigidTransform() = default;

const DOMPointReadOnly& WebXRRigidTransform::position() const
{
    return m_position;
}

const DOMPointReadOnly& WebXRRigidTransform::orientation() const
{
    return m_orientation;
}

const Float32Array& WebXRRigidTransform::matrix()
{
    if (m_matrix && !m_matrix->isDetached())
        return *m_matrix;

    // Lazily create matrix Float32Array.
    auto matrixData = m_rawTransform.toColumnMajorFloatArray();
    m_matrix = Float32Array::create(matrixData.data(), matrixData.size());

    return *m_matrix;
}

const WebXRRigidTransform& WebXRRigidTransform::inverse()
{
    // The inverse of a inverse object should return the original object.
    if (m_parentInverse)
        return *m_parentInverse;

    // Inverse should always return the same object.
    if (m_inverse)
        return *m_inverse;
    
    auto inverseTransform = m_rawTransform.inverse();
    ASSERT(!!inverseTransform);

    m_inverse = WebXRRigidTransform::create(*inverseTransform);
    // The inverse of a inverse object should return the original object.
    m_inverse->m_parentInverse = *this;

    return *m_inverse;
}

const TransformationMatrix& WebXRRigidTransform::rawTransform() const
{
    return m_rawTransform;
}

} // namespace WebCore

#endif // ENABLE(WEBXR)
