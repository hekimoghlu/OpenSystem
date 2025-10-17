/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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

class SkewTransformOperation final : public TransformOperation {
public:
    WEBCORE_EXPORT static Ref<SkewTransformOperation> create(double, double, TransformOperation::Type);

    Ref<TransformOperation> clone() const override
    {
        return adoptRef(*new SkewTransformOperation(m_angleX, m_angleY, type()));
    }

    double angleX() const { return m_angleX; }
    double angleY() const { return m_angleY; }

private:
    bool isIdentity() const override { return m_angleX == 0 && m_angleY == 0; }
    bool isAffectedByTransformOrigin() const override { return !isIdentity(); }

    bool operator==(const SkewTransformOperation& other) const { return operator==(static_cast<const TransformOperation&>(other)); }
    bool operator==(const TransformOperation&) const override;

    bool apply(TransformationMatrix& transform, const FloatSize&) const override
    {
        transform.skew(m_angleX, m_angleY);
        return false;
    }

    Ref<TransformOperation> blend(const TransformOperation* from, const BlendingContext&, bool blendToIdentity = false) override;

    void dump(WTF::TextStream&) const final;
    
    SkewTransformOperation(double, double, TransformOperation::Type);

    double m_angleX;
    double m_angleY;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_TRANSFORMOPERATION(WebCore::SkewTransformOperation, WebCore::TransformOperation::isSkewTransformOperationType)
