/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

class CSSStyleRule;
class StyleRule;

// https://drafts.css-houdini.org/css-typed-om/#declared-stylepropertymap-objects
class DeclaredStylePropertyMap final : public StylePropertyMap {
public:
    static Ref<DeclaredStylePropertyMap> create(CSSStyleRule&);

    Vector<StylePropertyMapEntry> entries(ScriptExecutionContext*) const final;
    unsigned size() const final;
    Type type() const final { return Type::Declared; }

private:
    explicit DeclaredStylePropertyMap(CSSStyleRule&);

    StyleRule* styleRule() const;

    void clear() final;
    RefPtr<CSSValue> propertyValue(CSSPropertyID) const final;
    String shorthandPropertySerialization(CSSPropertyID) const final;
    RefPtr<CSSValue> customPropertyValue(const AtomString&) const final;
    void removeProperty(CSSPropertyID) final;
    void removeCustomProperty(const AtomString&) final;
    bool setShorthandProperty(CSSPropertyID, const String&) final;
    bool setProperty(CSSPropertyID, Ref<CSSValue>&&) final;
    bool setCustomProperty(Document&, const AtomString&, Ref<CSSVariableReferenceValue>&&) final;

    WeakPtr<CSSStyleRule> m_ownerRule;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSSOM_STYLE_PROPERTY_MAP(DeclaredStylePropertyMap, WebCore::StylePropertyMapReadOnly::Type::Declared);
