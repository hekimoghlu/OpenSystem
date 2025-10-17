/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 28, 2022.
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

#include "TransformOperation.h"
#include <wtf/Ref.h>

namespace WebCore {

struct BlendingContext;

class RotateTransformOperation final : public TransformOperation {
public:
    static Ref<RotateTransformOperation> create(double angle, TransformOperation::Type type)
    {
        return adoptRef(*new RotateTransformOperation(0, 0, 1, angle, type));
    }

    WEBCORE_EXPORT static Ref<RotateTransformOperation> create(double, double, double, double, TransformOperation::Type);

    Ref<TransformOperation> clone() const final
    {
        return adoptRef(*new RotateTransformOperation(m_x, m_y, m_z, m_angle, type()));
    }

    double x() const { return m_x; }
    double y() const { return m_y; }
    double z() const { return m_z; }
    double angle() const { return m_angle; }

    TransformOperation::Type primitiveType() const final { return type() == Type::Rotate ? Type::Rotate : Type::Rotate3D; }

    bool operator==(const RotateTransformOperation& other) const { return operator==(static_cast<const TransformOperation&>(other)); }
    bool operator==(const TransformOperation&) const final;

    Ref<TransformOperation> blend(const TransformOperation* from, const BlendingContext&, bool blendToIdentity = false) final;

    bool isIdentity() const final { return !m_angle; }

    bool isRepresentableIn2D() const final { return (!m_x && !m_y) || !m_angle; }

    bool isAffectedByTransformOrigin() const final { return !isIdentity(); }

    bool apply(TransformationMatrix& transform, const FloatSize& /*borderBoxSize*/) const final
    {
        if (type() == TransformOperation::Type::Rotate)
            transform.rotate(m_angle);
        else
            transform.rotate3d(m_x, m_y, m_z, m_angle);
        return false;
    }

    bool applyUnrounded(TransformationMatrix& transform, const FloatSize& /*borderBoxSize*/) const final
    {
        if (type() == TransformOperation::Type::Rotate)
            transform.rotate(m_angle, TransformationMatrix::RotationSnapping::None);
        else
            transform.rotate3d(m_x, m_y, m_z, m_angle, TransformationMatrix::RotationSnapping::None);
        return false;
    }


    void dump(WTF::TextStream&) const final;

private:
    RotateTransformOperation(double, double, double, double, TransformOperation::Type);

    double m_x;
    double m_y;
    double m_z;
    double m_angle;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_TRANSFORMOPERATION(WebCore::RotateTransformOperation, WebCore::TransformOperation::isRotateTransformOperationType)
