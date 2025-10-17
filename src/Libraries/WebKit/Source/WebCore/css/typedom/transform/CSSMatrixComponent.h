/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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

#include "CSSMatrixComponentOptions.h"
#include "CSSTransformComponent.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class CSSFunctionValue;
class DOMMatrixReadOnly;
template<typename> class ExceptionOr;

class CSSMatrixComponent : public CSSTransformComponent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSMatrixComponent);
public:
    static Ref<CSSTransformComponent> create(Ref<DOMMatrixReadOnly>&&, CSSMatrixComponentOptions&& = { });
    static ExceptionOr<Ref<CSSTransformComponent>> create(Ref<const CSSFunctionValue>);

    DOMMatrix& matrix();
    void setMatrix(Ref<DOMMatrix>&&);

    void serialize(StringBuilder&) const final;
    ExceptionOr<Ref<DOMMatrix>> toMatrix() final;
    
    CSSTransformType getType() const final { return CSSTransformType::MatrixComponent; }

    RefPtr<CSSValue> toCSSValue() const final;

private:
    CSSMatrixComponent(Ref<DOMMatrixReadOnly>&&, Is2D);
    Ref<DOMMatrix> m_matrix;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CSSMatrixComponent)
    static bool isType(const WebCore::CSSTransformComponent& transform) { return transform.getType() == WebCore::CSSTransformType::MatrixComponent; }
SPECIALIZE_TYPE_TRAITS_END()
