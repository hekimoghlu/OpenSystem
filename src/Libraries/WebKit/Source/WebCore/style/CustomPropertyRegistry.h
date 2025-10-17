/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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

#include "CSSRegisteredCustomProperty.h"
#include "StyleRule.h"
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class RenderStyle;

namespace Style {

class Scope;

class CustomPropertyRegistry {
    WTF_MAKE_TZONE_ALLOCATED(CustomPropertyRegistry);
public:
    CustomPropertyRegistry(Scope&);

    const CSSRegisteredCustomProperty* get(const AtomString&) const;
    bool isInherited(const AtomString&) const;

    bool registerFromAPI(CSSRegisteredCustomProperty&&);
    void registerFromStylesheet(const StyleRuleProperty::Descriptor&);
    void clearRegisteredFromStylesheets();

    const RenderStyle& initialValuePrototypeStyle() const;

    bool invalidatePropertiesWithViewportUnits(Document&);

    enum class ViewportUnitDependency : bool { No, Yes };
    enum class ParseInitialValueError : uint8_t { NotComputationallyIndependent, DidNotParse };
    static Expected<std::pair<RefPtr<CSSCustomPropertyValue>, ViewportUnitDependency>, ParseInitialValueError> parseInitialValue(const Document&, const AtomString& propertyName, const CSSCustomPropertySyntax&, CSSParserTokenRange);

private:
    void invalidate(const AtomString&);
    void notifyAnimationsOfCustomPropertyRegistration(const AtomString&);

    Scope& m_scope;

    UncheckedKeyHashMap<AtomString, UniqueRef<CSSRegisteredCustomProperty>> m_propertiesFromAPI;
    UncheckedKeyHashMap<AtomString, UniqueRef<CSSRegisteredCustomProperty>> m_propertiesFromStylesheet;

    mutable std::unique_ptr<RenderStyle> m_initialValuePrototypeStyle;
    mutable bool m_hasInvalidPrototypeStyle { false };
};

}
}
