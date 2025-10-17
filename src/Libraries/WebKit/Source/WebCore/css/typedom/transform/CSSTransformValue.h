/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 22, 2022.
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

#include "CSSStyleValue.h"
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CSSTransformComponent;
class CSSTransformListValue;
class DOMMatrix;
template<typename> class ExceptionOr;

class CSSTransformValue final : public CSSStyleValue {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSTransformValue);
public:
    static ExceptionOr<Ref<CSSTransformValue>> create(Ref<const CSSTransformListValue>);
    static ExceptionOr<Ref<CSSTransformValue>> create(Vector<Ref<CSSTransformComponent>>&&);

    virtual ~CSSTransformValue();

    size_t length() const { return m_components.size(); }
    bool isSupportedPropertyIndex(unsigned index) const { return index < m_components.size(); }
    RefPtr<CSSTransformComponent> item(size_t);
    ExceptionOr<Ref<CSSTransformComponent>> setItem(size_t, Ref<CSSTransformComponent>&&);
    
    bool is2D() const;
    
    ExceptionOr<Ref<DOMMatrix>> toMatrix();
    
    CSSStyleValueType getType() const override { return CSSStyleValueType::CSSTransformValue; }

    RefPtr<CSSValue> toCSSValue() const final;

private:
    CSSTransformValue(Vector<Ref<CSSTransformComponent>>&&);
    void serialize(StringBuilder&, OptionSet<SerializationArguments>) const final;

    Vector<Ref<CSSTransformComponent>> m_components;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CSSTransformValue)
    static bool isType(const WebCore::CSSStyleValue& styleValue) { return styleValue.getType() == WebCore::CSSStyleValueType::CSSTransformValue; }
SPECIALIZE_TYPE_TRAITS_END()
