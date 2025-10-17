/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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

class CSSTranslate : public CSSTransformComponent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSTranslate);
public:
    static ExceptionOr<Ref<CSSTranslate>> create(Ref<CSSNumericValue> x, Ref<CSSNumericValue> y, RefPtr<CSSNumericValue> z);
    static ExceptionOr<Ref<CSSTranslate>> create(Ref<const CSSFunctionValue>);

    const CSSNumericValue& x() const { return m_x.get(); }
    const CSSNumericValue& y() const { return m_y.get(); }
    const CSSNumericValue& z() const { return m_z.get(); }

    void setX(Ref<CSSNumericValue> x) { m_x = WTFMove(x); }
    void setY(Ref<CSSNumericValue> y) { m_y = WTFMove(y); }
    ExceptionOr<void> setZ(Ref<CSSNumericValue>);
    
    void serialize(StringBuilder&) const final;
    ExceptionOr<Ref<DOMMatrix>> toMatrix() final;

    RefPtr<CSSValue> toCSSValue() const final;

private:
    CSSTranslate(CSSTransformComponent::Is2D, Ref<CSSNumericValue>, Ref<CSSNumericValue>, Ref<CSSNumericValue>);

    CSSTransformType getType() const final { return CSSTransformType::Translate; }

    Ref<CSSNumericValue> m_x;
    Ref<CSSNumericValue> m_y;
    Ref<CSSNumericValue> m_z;
};
    
} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CSSTranslate)
    static bool isType(const WebCore::CSSTransformComponent& transform) { return transform.getType() == WebCore::CSSTransformType::Translate; }
SPECIALIZE_TYPE_TRAITS_END()
