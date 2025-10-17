/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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

class ScaleTransformOperation final : public TransformOperation {
public:
    static Ref<ScaleTransformOperation> create(double sx, double sy, TransformOperation::Type type)
    {
        return adoptRef(*new ScaleTransformOperation(sx, sy, 1, type));
    }

    WEBCORE_EXPORT static Ref<ScaleTransformOperation> create(double, double, double, TransformOperation::Type);

    Ref<TransformOperation> clone() const final
    {
        return adoptRef(*new ScaleTransformOperation(m_x, m_y, m_z, type()));
    }

    double x() const { return m_x; }
    double y() const { return m_y; }
    double z() const { return m_z; }

    TransformOperation::Type primitiveType() const final { return (type() == Type::ScaleZ || type() == Type::Scale3D) ? Type::Scale3D : Type::Scale; }

    bool operator==(const ScaleTransformOperation& other) const { return operator==(static_cast<const TransformOperation&>(other)); }
    bool operator==(const TransformOperation&) const final;

    Ref<TransformOperation> blend(const TransformOperation* from, const BlendingContext&, bool blendToIdentity = false) final;

    bool isIdentity() const final { return m_x == 1 &&  m_y == 1 &&  m_z == 1; }

    bool isRepresentableIn2D() const final { return m_z == 1; }

    bool isAffectedByTransformOrigin() const final { return !isIdentity(); }

    bool apply(TransformationMatrix& transform, const FloatSize&) const final
    {
        transform.scale3d(m_x, m_y, m_z);
        return false;
    }

    void dump(WTF::TextStream&) const final;

private:
    ScaleTransformOperation(double, double, double, TransformOperation::Type);

    double m_x;
    double m_y;
    double m_z;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_TRANSFORMOPERATION(WebCore::ScaleTransformOperation, WebCore::TransformOperation::isScaleTransformOperationType)
