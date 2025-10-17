/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 5, 2024.
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
#include <optional>
#include <wtf/Ref.h>

namespace WebCore {

struct BlendingContext;

class PerspectiveTransformOperation final : public TransformOperation {
public:
    WEBCORE_EXPORT static Ref<PerspectiveTransformOperation> create(const std::optional<Length>&);

    Ref<TransformOperation> clone() const override
    {
        return adoptRef(*new PerspectiveTransformOperation(m_p));
    }

    std::optional<Length> perspective() const { return m_p; }
    
private:
    bool isIdentity() const override { return !m_p; }
    bool isAffectedByTransformOrigin() const override { return !isIdentity(); }
    bool isRepresentableIn2D() const final { return false; }

    bool operator==(const PerspectiveTransformOperation& other) const { return operator==(static_cast<const TransformOperation&>(other)); }
    bool operator==(const TransformOperation&) const override;

    std::optional<float> floatValue() const
    {
        if (!m_p)
            return { };

        // From https://www.w3.org/TR/css-transforms-2/#perspective-property:
        // "As very small <length> values can produce bizarre rendering results and stress the numerical accuracy of
        // transform calculations, values less than 1px must be treated as 1px for rendering purposes. (This clamping
        // does not affect the underlying value, so perspective: 0; in a stylesheet will still serialize back as 0.)"
        return std::max(1.0f, floatValueForLength(*m_p, 1.0));
    }

    bool apply(TransformationMatrix& transform, const FloatSize&) const override
    {
        if (auto value = floatValue())
            transform.applyPerspective(*value);
        return false;
    }

    Ref<TransformOperation> blend(const TransformOperation* from, const BlendingContext&, bool blendToIdentity = false) override;

    void dump(WTF::TextStream&) const final;

    PerspectiveTransformOperation(const std::optional<Length>&);

    std::optional<Length> m_p;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_TRANSFORMOPERATION(WebCore::PerspectiveTransformOperation, WebCore::TransformOperation::Type::Perspective ==)
