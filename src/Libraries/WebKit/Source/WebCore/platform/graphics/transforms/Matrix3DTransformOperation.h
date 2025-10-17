/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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

class Matrix3DTransformOperation final : public TransformOperation {
public:
    WEBCORE_EXPORT static Ref<Matrix3DTransformOperation> create(const TransformationMatrix&);

    Ref<TransformOperation> clone() const override
    {
        return adoptRef(*new Matrix3DTransformOperation(m_matrix));
    }

    TransformationMatrix matrix() const {return m_matrix; }

private:    
    bool isIdentity() const override { return m_matrix.isIdentity(); }
    bool isAffectedByTransformOrigin() const final { return !isIdentity(); }

    bool isRepresentableIn2D() const final;

    bool operator==(const Matrix3DTransformOperation& other) const { return operator==(static_cast<const TransformOperation&>(other)); }
    bool operator==(const TransformOperation&) const override;

    bool apply(TransformationMatrix& transform, const FloatSize&) const override
    {
        transform.multiply(m_matrix);
        return false;
    }

    Ref<TransformOperation> blend(const TransformOperation* from, const BlendingContext&, bool blendToIdentity = false) override;
    
    void dump(WTF::TextStream&) const final;

    Matrix3DTransformOperation(const TransformationMatrix&);

    TransformationMatrix m_matrix;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_TRANSFORMOPERATION(WebCore::Matrix3DTransformOperation, WebCore::TransformOperation::Type::Matrix3D ==)
