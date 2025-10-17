/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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

class CSSScale : public CSSTransformComponent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSScale);
public:
    static ExceptionOr<Ref<CSSScale>> create(CSSNumberish x, CSSNumberish y, std::optional<CSSNumberish>&& z);
    static ExceptionOr<Ref<CSSScale>> create(Ref<const CSSFunctionValue>);

    void serialize(StringBuilder&) const final;
    ExceptionOr<Ref<DOMMatrix>> toMatrix() final;

    CSSNumberish x() const { return m_x.ptr(); }
    CSSNumberish y() const { return m_y.ptr(); }
    CSSNumberish z() const { return m_z.ptr(); }

    void setX(CSSNumberish);
    void setY(CSSNumberish);
    void setZ(CSSNumberish);

    CSSTransformType getType() const final { return CSSTransformType::Scale; }

    RefPtr<CSSValue> toCSSValue() const final;

private:
    CSSScale(CSSTransformComponent::Is2D, Ref<CSSNumericValue>, Ref<CSSNumericValue>, Ref<CSSNumericValue>);

    Ref<CSSNumericValue> m_x;
    Ref<CSSNumericValue> m_y;
    Ref<CSSNumericValue> m_z;
};
    
} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CSSScale)
    static bool isType(const WebCore::CSSTransformComponent& transform) { return transform.getType() == WebCore::CSSTransformType::Scale; }
SPECIALIZE_TYPE_TRAITS_END()
