/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 19, 2024.
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

#include "StylePropertyMap.h"
#include <wtf/WeakPtr.h>

namespace WebCore {

class WeakPtrImplWithEventTargetData;

class InlineStylePropertyMap final : public StylePropertyMap {
public:
    static Ref<InlineStylePropertyMap> create(StyledElement&);

    Type type() const final { return Type::Inline; }

    RefPtr<CSSValue> propertyValue(CSSPropertyID) const final;
    String shorthandPropertySerialization(CSSPropertyID) const final;
    RefPtr<CSSValue> customPropertyValue(const AtomString& property) const final;
    unsigned size() const final;
    Vector<StylePropertyMapEntry> entries(ScriptExecutionContext*) const final;
    void removeProperty(CSSPropertyID) final;
    bool setShorthandProperty(CSSPropertyID, const String& value) final;
    bool setProperty(CSSPropertyID, Ref<CSSValue>&&) final;
    bool setCustomProperty(Document&, const AtomString& property, Ref<CSSVariableReferenceValue>&&) final;
    void removeCustomProperty(const AtomString& property) final;
    void clear() final;

private:
    explicit InlineStylePropertyMap(StyledElement&);

    WeakPtr<StyledElement, WeakPtrImplWithEventTargetData> m_element;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSSOM_STYLE_PROPERTY_MAP(InlineStylePropertyMap, WebCore::StylePropertyMapReadOnly::Type::Inline);
