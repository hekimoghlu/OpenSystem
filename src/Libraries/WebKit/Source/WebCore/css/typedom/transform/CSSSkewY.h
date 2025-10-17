/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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

class CSSSkewY : public CSSTransformComponent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSSkewY);
public:
    static ExceptionOr<Ref<CSSSkewY>> create(Ref<CSSNumericValue>);
    static ExceptionOr<Ref<CSSSkewY>> create(Ref<const CSSFunctionValue>);

    const CSSNumericValue& ay() const { return m_ay.get(); }
    ExceptionOr<void> setAy(Ref<CSSNumericValue>);

    void serialize(StringBuilder&) const final;
    ExceptionOr<Ref<DOMMatrix>> toMatrix() final;
    void setIs2D(bool) final { };

    CSSTransformType getType() const final { return CSSTransformType::SkewY; }

    RefPtr<CSSValue> toCSSValue() const final;

private:
    CSSSkewY(Ref<CSSNumericValue> ay);
    
    Ref<CSSNumericValue> m_ay;
};
    
} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CSSSkewY)
    static bool isType(const WebCore::CSSTransformComponent& transform) { return transform.getType() == WebCore::CSSTransformType::SkewY; }
SPECIALIZE_TYPE_TRAITS_END()
