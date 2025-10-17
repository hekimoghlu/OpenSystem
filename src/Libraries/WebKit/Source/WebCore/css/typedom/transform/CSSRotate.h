/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 17, 2023.
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

#include "CSSNumericValue.h"
#include "CSSTransformComponent.h"

namespace WebCore {

class CSSFunctionValue;

template<typename> class ExceptionOr;

class CSSRotate : public CSSTransformComponent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSRotate);
public:
    static ExceptionOr<Ref<CSSRotate>> create(CSSNumberish, CSSNumberish, CSSNumberish, Ref<CSSNumericValue>);
    static ExceptionOr<Ref<CSSRotate>> create(Ref<CSSNumericValue>);
    static ExceptionOr<Ref<CSSRotate>> create(Ref<const CSSFunctionValue>);

    CSSNumberish x() { return { m_x.ptr() }; }
    CSSNumberish y() { return { m_y.ptr() }; }
    CSSNumberish z() { return { m_z.ptr() }; }
    const CSSNumericValue& angle() { return m_angle.get(); }

    ExceptionOr<void> setX(CSSNumberish);
    ExceptionOr<void> setY(CSSNumberish);
    ExceptionOr<void> setZ(CSSNumberish);
    ExceptionOr<void> setAngle(Ref<CSSNumericValue>);

    void serialize(StringBuilder&) const final;
    ExceptionOr<Ref<DOMMatrix>> toMatrix() final;
    
    CSSTransformType getType() const final { return CSSTransformType::Rotate; }

    RefPtr<CSSValue> toCSSValue() const final;
    
private:
    CSSRotate(CSSTransformComponent::Is2D, Ref<CSSNumericValue>, Ref<CSSNumericValue>, Ref<CSSNumericValue>, Ref<CSSNumericValue>);
    
    Ref<CSSNumericValue> m_x;
    Ref<CSSNumericValue> m_y;
    Ref<CSSNumericValue> m_z;
    Ref<CSSNumericValue> m_angle;
};
    
} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CSSRotate)
    static bool isType(const WebCore::CSSTransformComponent& transform) { return transform.getType() == WebCore::CSSTransformType::Rotate; }
SPECIALIZE_TYPE_TRAITS_END()
