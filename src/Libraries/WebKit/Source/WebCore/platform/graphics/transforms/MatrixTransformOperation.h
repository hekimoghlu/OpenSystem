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
#include "TransformationMatrix.h"
#include <wtf/Ref.h>

namespace WebCore {

struct BlendingContext;

class MatrixTransformOperation final : public TransformOperation {
public:
    static Ref<MatrixTransformOperation> create(double a, double b, double c, double d, double e, double f)
    {
        return adoptRef(*new MatrixTransformOperation(a, b, c, d, e, f));
    }

    WEBCORE_EXPORT static Ref<MatrixTransformOperation> create(const TransformationMatrix&);

    Ref<TransformOperation> clone() const override
    {
        return adoptRef(*new MatrixTransformOperation(matrix()));
    }

    TransformationMatrix matrix() const { return TransformationMatrix(m_a, m_b, m_c, m_d, m_e, m_f); }

private:
    bool isIdentity() const override { return m_a == 1 && m_b == 0 && m_c == 0 && m_d == 1 && m_e == 0 && m_f == 0; }
    bool isAffectedByTransformOrigin() const override { return !isIdentity(); }

    bool operator==(const MatrixTransformOperation& other) const { return operator==(static_cast<const TransformOperation&>(other)); }
    bool operator==(const TransformOperation&) const override;

    bool apply(TransformationMatrix& transform, const FloatSize&) const override
    {
        TransformationMatrix matrix(m_a, m_b, m_c, m_d, m_e, m_f);
        transform.multiply(matrix);
        return false;
    }

    Ref<TransformOperation> blend(const TransformOperation* from, const BlendingContext&, bool blendToIdentity = false) override;

    void dump(WTF::TextStream&) const final;

    MatrixTransformOperation(double a, double b, double c, double d, double e, double f)
        : TransformOperation(TransformOperation::Type::Matrix)
        , m_a(a)
        , m_b(b)
        , m_c(c)
        , m_d(d)
        , m_e(e)
        , m_f(f)
    {
    }

    MatrixTransformOperation(const TransformationMatrix&);

    double m_a;
    double m_b;
    double m_c;
    double m_d;
    double m_e;
    double m_f;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_TRANSFORMOPERATION(WebCore::MatrixTransformOperation, WebCore::TransformOperation::Type::Matrix ==)
