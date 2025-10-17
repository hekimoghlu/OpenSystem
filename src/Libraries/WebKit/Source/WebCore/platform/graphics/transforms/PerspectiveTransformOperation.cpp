/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 25, 2024.
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
#include "PerspectiveTransformOperation.h"

#include "AnimationUtilities.h"
#include <wtf/MathExtras.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<PerspectiveTransformOperation> PerspectiveTransformOperation::create(const std::optional<Length>& p)
{
    return adoptRef(*new PerspectiveTransformOperation(p));
}

PerspectiveTransformOperation::PerspectiveTransformOperation(const std::optional<Length>& p)
    : TransformOperation(TransformOperation::Type::Perspective)
    , m_p(p)
{
    ASSERT(!p || (*p).isFixed());
}

bool PerspectiveTransformOperation::operator==(const TransformOperation& other) const
{
    if (!isSameType(other))
        return false;
    return m_p == downcast<PerspectiveTransformOperation>(other).m_p;
}

Ref<TransformOperation> PerspectiveTransformOperation::blend(const TransformOperation* from, const BlendingContext& context, bool blendToIdentity)
{
    if (!sharedPrimitiveType(from))
        return *this;

    // https://drafts.csswg.org/css-transforms-2/#interpolation-of-transform-functions
    // says that we should run matrix decomposition and then run the rules for
    // interpolation of matrices, but we know what those rules are going to
    // yield, so just do that directly.
    auto getInverse = [](const auto& operation) {
        return !operation->isIdentity() ? (1.0 / (*operation->floatValue())) : 0.0;
    };

    double ourInverse = getInverse(this);
    double fromPInverse, toPInverse;
    if (blendToIdentity) {
        fromPInverse = ourInverse;
        toPInverse = 0.0;
    } else {
        fromPInverse = from ? getInverse(downcast<PerspectiveTransformOperation>(from)) : 0.0;
        toPInverse = ourInverse;
    }

    double pInverse = WebCore::blend(fromPInverse, toPInverse, context);
    std::optional<Length> p;
    if (pInverse > 0.0 && std::isnormal(pInverse)) {
        p = Length(1.0 / pInverse, LengthType::Fixed);
    }
    return PerspectiveTransformOperation::create(p);
}

void PerspectiveTransformOperation::dump(TextStream& ts) const
{
    ts << type() << "(";
    if (!m_p)
        ts << "none";
    else
        ts << m_p;
    ts << ")";
}

} // namespace WebCore
