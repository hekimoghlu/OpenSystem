/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 10, 2022.
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

#include "CSSKeywordValue.h"
#include "CSSNumericValue.h"
#include "CSSTransformComponent.h"

namespace WebCore {

template<typename> class ExceptionOr;
class CSSFunctionValue;
using CSSPerspectiveValue = std::variant<RefPtr<CSSNumericValue>, String, RefPtr<CSSKeywordValue>>;

class CSSPerspective : public CSSTransformComponent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSPerspective);
public:
    static ExceptionOr<Ref<CSSPerspective>> create(CSSPerspectiveValue);
    static ExceptionOr<Ref<CSSPerspective>> create(Ref<const CSSFunctionValue>);

    virtual ~CSSPerspective();

    const CSSPerspectiveValue& length() const { return m_length; }
    ExceptionOr<void> setLength(CSSPerspectiveValue);

    void serialize(StringBuilder&) const final;
    ExceptionOr<Ref<DOMMatrix>> toMatrix() final;
    
    CSSTransformType getType() const final { return CSSTransformType::Perspective; }

    RefPtr<CSSValue> toCSSValue() const final;

private:
    CSSPerspective(CSSPerspectiveValue);

    void setIs2D(bool);

    CSSPerspectiveValue m_length;
};
    
} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CSSPerspective)
    static bool isType(const WebCore::CSSTransformComponent& transform) { return transform.getType() == WebCore::CSSTransformType::Perspective; }
SPECIALIZE_TYPE_TRAITS_END()
