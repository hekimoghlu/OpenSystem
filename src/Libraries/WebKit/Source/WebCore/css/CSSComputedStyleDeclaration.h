/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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

#include "CSSStyleDeclaration.h"
#include "ComputedStyleExtractor.h"
#include "PseudoElementIdentifier.h"
#include "RenderStyleConstants.h"
#include <wtf/FixedVector.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Element;
class MutableStyleProperties;

class CSSComputedStyleDeclaration final : public CSSStyleDeclaration, public RefCounted<CSSComputedStyleDeclaration> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(CSSComputedStyleDeclaration, WEBCORE_EXPORT);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    enum class AllowVisited : bool { No, Yes };
    WEBCORE_EXPORT static Ref<CSSComputedStyleDeclaration> create(Element&, AllowVisited);
    static Ref<CSSComputedStyleDeclaration> create(Element&, const std::optional<Style::PseudoElementIdentifier>&);
    static Ref<CSSComputedStyleDeclaration> createEmpty(Element&);

    WEBCORE_EXPORT virtual ~CSSComputedStyleDeclaration();

    String getPropertyValue(CSSPropertyID) const;

private:
    enum class IsEmpty : bool { No, Yes };
    CSSComputedStyleDeclaration(Element&, AllowVisited);
    CSSComputedStyleDeclaration(Element&, IsEmpty);
    CSSComputedStyleDeclaration(Element&, const std::optional<Style::PseudoElementIdentifier>&);

    // CSSOM functions. Don't make these public.
    CSSRule* parentRule() const final;
    CSSRule* cssRules() const final;
    unsigned length() const final;
    String item(unsigned index) const final;
    RefPtr<DeprecatedCSSOMValue> getPropertyCSSValue(const String& propertyName) final;
    String getPropertyValue(const String& propertyName) final;
    String getPropertyPriority(const String& propertyName) final;
    String getPropertyShorthand(const String& propertyName) final;
    bool isPropertyImplicit(const String& propertyName) final;
    ExceptionOr<void> setProperty(const String& propertyName, const String& value, const String& priority) final;
    ExceptionOr<String> removeProperty(const String& propertyName) final;
    String cssText() const final;
    ExceptionOr<void> setCssText(const String&) final;
    String getPropertyValueInternal(CSSPropertyID) final;
    ExceptionOr<void> setPropertyInternal(CSSPropertyID, const String& value, IsImportant) final;
    Ref<MutableStyleProperties> copyProperties() const final;

    RefPtr<CSSValue> getPropertyCSSValue(CSSPropertyID, ComputedStyleExtractor::UpdateLayout = ComputedStyleExtractor::UpdateLayout::Yes) const;

    const Settings* settings() const final;
    const FixedVector<CSSPropertyID>& exposedComputedCSSPropertyIDs() const;

    mutable Ref<Element> m_element;
    std::optional<Style::PseudoElementIdentifier> m_pseudoElementIdentifier { std::nullopt };
    bool m_isEmpty { false };
    bool m_allowVisitedStyle { false };
};

} // namespace WebCore
