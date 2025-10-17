/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 20, 2024.
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

#include "Length.h"
#include "LengthFunctions.h"
#include "TransformOperation.h"
#include <wtf/Ref.h>

namespace WebCore {

struct BlendingContext;

class TranslateTransformOperation final : public TransformOperation {
public:
    static Ref<TranslateTransformOperation> create(const Length& tx, const Length& ty, TransformOperation::Type type)
    {
        return adoptRef(*new TranslateTransformOperation(tx, ty, Length(0, LengthType::Fixed), type));
    }

    WEBCORE_EXPORT static Ref<TranslateTransformOperation> create(const Length&, const Length&, const Length&, TransformOperation::Type);

    Ref<TransformOperation> clone() const override
    {
        return adoptRef(*new TranslateTransformOperation(m_x, m_y, m_z, type()));
    }

    Ref<TransformOperation> selfOrCopyWithResolvedCalculatedValues(const FloatSize&) override;

    float xAsFloat(const FloatSize& borderBoxSize) const { return floatValueForLength(m_x, borderBoxSize.width()); }
    float yAsFloat(const FloatSize& borderBoxSize) const { return floatValueForLength(m_y, borderBoxSize.height()); }
    float zAsFloat() const { return floatValueForLength(m_z, 1); }

    Length x() const { return m_x; }
    Length y() const { return m_y; }
    Length z() const { return m_z; }

    void setX(Length newX) { m_x = newX; }
    void setY(Length newY) { m_y = newY; }
    void setZ(Length newZ) { m_z = newZ; }

    TransformOperation::Type primitiveType() const final { return isRepresentableIn2D() ? Type::Translate : Type::Translate3D; }

    bool apply(TransformationMatrix& transform, const FloatSize& borderBoxSize) const final
    {
        transform.translate3d(xAsFloat(borderBoxSize), yAsFloat(borderBoxSize), zAsFloat());
        return m_x.isPercent() || m_y.isPercent();
    }

    bool isIdentity() const final { return !floatValueForLength(m_x, 1) && !floatValueForLength(m_y, 1) && !floatValueForLength(m_z, 1); }

    bool operator==(const TranslateTransformOperation& other) const { return operator==(static_cast<const TransformOperation&>(other)); }
    bool operator==(const TransformOperation&) const final;

    Ref<TransformOperation> blend(const TransformOperation* from, const BlendingContext&, bool blendToIdentity = false) final;

    bool isRepresentableIn2D() const final { return m_z.isZero(); }

private:

    void dump(WTF::TextStream&) const final;

    TranslateTransformOperation(const Length&, const Length&, const Length&, TransformOperation::Type);

    Length m_x;
    Length m_y;
    Length m_z;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_TRANSFORMOPERATION(WebCore::TranslateTransformOperation, WebCore::TransformOperation::isTranslateTransformOperationType)
