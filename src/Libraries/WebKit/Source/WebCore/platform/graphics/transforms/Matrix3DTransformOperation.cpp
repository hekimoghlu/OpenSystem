/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 5, 2025.
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
#include "Matrix3DTransformOperation.h"

#include "AnimationUtilities.h"
#include <algorithm>
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<Matrix3DTransformOperation> Matrix3DTransformOperation::create(const TransformationMatrix& matrix)
{
    return adoptRef(*new Matrix3DTransformOperation(matrix));
}

Matrix3DTransformOperation::Matrix3DTransformOperation(const TransformationMatrix& mat)
    : TransformOperation(TransformOperation::Type::Matrix3D)
    , m_matrix(mat)
{
}

bool Matrix3DTransformOperation::operator==(const TransformOperation& other) const
{
    return isSameType(other) && m_matrix == downcast<Matrix3DTransformOperation>(other).m_matrix;
}

static Ref<TransformOperation> createOperation(TransformationMatrix& to, TransformationMatrix& from, const BlendingContext& context)
{
    to.blend(from, context.progress, context.compositeOperation);
    return Matrix3DTransformOperation::create(to);
}

Ref<TransformOperation> Matrix3DTransformOperation::blend(const TransformOperation* from, const BlendingContext& context, bool blendToIdentity)
{
    if (!sharedPrimitiveType(from))
        return *this;

    // Convert the TransformOperations into matrices
    FloatSize size;
    TransformationMatrix fromT;
    TransformationMatrix toT;
    if (from)
        from->apply(fromT, size);

    apply(toT, size);

    if (blendToIdentity)
        return createOperation(fromT, toT, context);
    return createOperation(toT, fromT, context);
}

bool Matrix3DTransformOperation::isRepresentableIn2D() const
{
    return m_matrix.isAffine();
}

void Matrix3DTransformOperation::dump(TextStream& ts) const
{
    ts << type() << "(" << m_matrix << ")";
}

} // namespace WebCore
