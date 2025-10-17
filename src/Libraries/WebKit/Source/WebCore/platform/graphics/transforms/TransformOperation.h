/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 4, 2025.
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

#include "CompositeOperation.h"
#include "FloatSize.h"
#include "TransformationMatrix.h"
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/TypeCasts.h>

namespace WebCore {

struct BlendingContext;

enum class TransformOperationType : uint8_t {
    ScaleX,
    ScaleY,
    Scale,
    TranslateX,
    TranslateY,
    Translate,
    RotateX,
    RotateY,
    Rotate,
    SkewX,
    SkewY,
    Skew,
    Matrix,
    ScaleZ,
    Scale3D,
    TranslateZ,
    Translate3D,
    RotateZ,
    Rotate3D,
    Matrix3D,
    Perspective,
    Identity,
    None
};

class TransformOperation : public RefCounted<TransformOperation> {
public:
    using Type = TransformOperationType;

    TransformOperation(Type type)
        : m_type(type)
    {
    }
    virtual ~TransformOperation() = default;

    virtual Ref<TransformOperation> clone() const = 0;
    virtual Ref<TransformOperation> selfOrCopyWithResolvedCalculatedValues(const FloatSize&) { return *this; }

    virtual bool operator==(const TransformOperation&) const = 0;

    virtual bool isIdentity() const = 0;

    // Return true if the borderBoxSize was used in the computation, false otherwise.
    virtual bool apply(TransformationMatrix&, const FloatSize& borderBoxSize) const = 0;
    virtual bool applyUnrounded(TransformationMatrix& transform, const FloatSize& borderBoxSize) const
    {
        return apply(transform, borderBoxSize);
    }

    virtual Ref<TransformOperation> blend(const TransformOperation* from, const BlendingContext&, bool blendToIdentity = false) = 0;

    Type type() const { return m_type; }
    bool isSameType(const TransformOperation& other) const { return type() == other.type(); }

    virtual Type primitiveType() const { return m_type; }
    std::optional<Type> sharedPrimitiveType(Type other) const;
    std::optional<Type> sharedPrimitiveType(const TransformOperation* other) const;

    virtual bool isAffectedByTransformOrigin() const { return false; }
    
    bool is3DOperation() const
    {
        Type opType = type();
        return opType == Type::ScaleZ
            || opType == Type::Scale3D
            || opType == Type::TranslateZ
            || opType == Type::Translate3D
            || opType == Type::RotateX
            || opType == Type::RotateY
            || opType == Type::Rotate3D
            || opType == Type::Matrix3D
            || opType == Type::Perspective;
    }
    
    virtual bool isRepresentableIn2D() const { return true; }

    static bool isRotateTransformOperationType(Type type)
    {
        return type == Type::RotateX
            || type == Type::RotateY
            || type == Type::RotateZ
            || type == Type::Rotate
            || type == Type::Rotate3D;
    }

    static bool isScaleTransformOperationType(Type type)
    {
        return type == Type::ScaleX
            || type == Type::ScaleY
            || type == Type::ScaleZ
            || type == Type::Scale
            || type == Type::Scale3D;
    }

    static bool isSkewTransformOperationType(Type type)
    {
        return type == Type::SkewX
            || type == Type::SkewY
            || type == Type::Skew;
    }

    static bool isTranslateTransformOperationType(Type type)
    {
        return type == Type::TranslateX
            || type == Type::TranslateY
            || type == Type::TranslateZ
            || type == Type::Translate
            || type == Type::Translate3D;
    }
    
    virtual void dump(WTF::TextStream&) const = 0;

private:
    Type m_type;
};

WTF::TextStream& operator<<(WTF::TextStream&, TransformOperation::Type);
WTF::TextStream& operator<<(WTF::TextStream&, const TransformOperation&);

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_TRANSFORMOPERATION(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(ToValueTypeName) \
    static bool isType(const WebCore::TransformOperation& operation) { return predicate(operation.type()); } \
SPECIALIZE_TYPE_TRAITS_END()
