/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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
#include "TransformOperation.h"

#include "IdentityTransformOperation.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

void IdentityTransformOperation::dump(TextStream& ts) const
{
    ts << type();
}

TextStream& operator<<(TextStream& ts, TransformOperation::Type type)
{
    switch (type) {
    case TransformOperation::Type::ScaleX: ts << "scaleX"; break;
    case TransformOperation::Type::ScaleY: ts << "scaleY"; break;
    case TransformOperation::Type::Scale: ts << "scale"; break;
    case TransformOperation::Type::TranslateX: ts << "translateX"; break;
    case TransformOperation::Type::TranslateY: ts << "translateY"; break;
    case TransformOperation::Type::Translate: ts << "translate"; break;
    case TransformOperation::Type::Rotate: ts << "rotate"; break;
    case TransformOperation::Type::SkewX: ts << "skewX"; break;
    case TransformOperation::Type::SkewY: ts << "skewY"; break;
    case TransformOperation::Type::Skew: ts << "skew"; break;
    case TransformOperation::Type::Matrix: ts << "matrix"; break;
    case TransformOperation::Type::ScaleZ: ts << "scaleX"; break;
    case TransformOperation::Type::Scale3D: ts << "scale3d"; break;
    case TransformOperation::Type::TranslateZ: ts << "translateZ"; break;
    case TransformOperation::Type::Translate3D: ts << "translate3d"; break;
    case TransformOperation::Type::RotateX: ts << "rotateX"; break;
    case TransformOperation::Type::RotateY: ts << "rotateY"; break;
    case TransformOperation::Type::RotateZ: ts << "rotateZ"; break;
    case TransformOperation::Type::Rotate3D: ts << "rotate3d"; break;
    case TransformOperation::Type::Matrix3D: ts << "matrix3d"; break;
    case TransformOperation::Type::Perspective: ts << "perspective"; break;
    case TransformOperation::Type::Identity: ts << "identity"; break;
    case TransformOperation::Type::None: ts << "none"; break;
    }
    
    return ts;
}

TextStream& operator<<(TextStream& ts, const TransformOperation& operation)
{
    operation.dump(ts);
    return ts;
}

std::optional<TransformOperation::Type> TransformOperation::sharedPrimitiveType(Type other) const
{
    // https://drafts.csswg.org/css-transforms-2/#interpolation-of-transform-functions
    // "If both transform functions share a primitive in the two-dimensional space, both transform
    // functions get converted to the two-dimensional primitive. If one or both transform functions
    // are three-dimensional transform functions, the common three-dimensional primitive is used."
    auto type = primitiveType();
    if (type == other)
        return type;
    static constexpr std::array sharedPrimitives {
        std::array { Type::Rotate, Type::Rotate3D },
        std::array { Type::Scale, Type::Scale3D },
        std::array { Type::Translate, Type::Translate3D }
    };
    for (auto typePair : sharedPrimitives) {
        if ((type == typePair[0] || type == typePair[1]) && (other == typePair[0] || other == typePair[1]))
            return typePair[1];
    }
    return std::nullopt;
}

std::optional<TransformOperation::Type> TransformOperation::sharedPrimitiveType(const TransformOperation* other) const
{
    // Blending with a null operation is always supported via blending with identity.
    if (!other)
        return type();

    // In case we have the same type, make sure to preserve it.
    if (other->type() == type())
        return type();

    return sharedPrimitiveType(other->primitiveType());
}

} // namespace WebCore
