/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 27, 2023.
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

#include <wtf/RobinHoodHashSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebCore {

class QualifiedName;
class SVGElement;
class SVGStringList;
class WeakPtrImplWithEventTargetData;

template<typename OwnerType, typename... BaseTypes>
class SVGPropertyOwnerRegistry;

class SVGTests;

class SVGConditionalProcessingAttributes {
    WTF_MAKE_TZONE_ALLOCATED(SVGConditionalProcessingAttributes);
    WTF_MAKE_NONCOPYABLE(SVGConditionalProcessingAttributes);
public:
    SVGConditionalProcessingAttributes(SVGElement& contextElement);

    SVGStringList& requiredExtensions() { return m_requiredExtensions; }
    SVGStringList& systemLanguage() { return m_systemLanguage; }

private:
    Ref<SVGStringList> m_requiredExtensions;
    Ref<SVGStringList> m_systemLanguage;
};

class SVGTests {
    WTF_MAKE_NONCOPYABLE(SVGTests);
public:
    static bool hasExtension(const String&);
    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGTests>;

    static void addSupportedAttributes(MemoryCompactLookupOnlyRobinHoodHashSet<QualifiedName>&);

    SVGConditionalProcessingAttributes& conditionalProcessingAttributes();
    SVGConditionalProcessingAttributes* conditionalProcessingAttributesIfExists() const;

    // These methods are called from DOM through the super classes.
    SVGStringList& requiredExtensions() { return conditionalProcessingAttributes().requiredExtensions(); }
    Ref<SVGStringList> protectedRequiredExtensions();
    SVGStringList& systemLanguage() { return conditionalProcessingAttributes().systemLanguage(); }
    Ref<SVGStringList> protectedSystemLanguage();

protected:
    bool isValid() const;

    void parseAttribute(const QualifiedName&, const AtomString&);
    void svgAttributeChanged(const QualifiedName&);

    SVGTests(SVGElement* contextElement);

private:
    Ref<SVGElement> protectedContextElement() const;

    WeakRef<SVGElement, WeakPtrImplWithEventTargetData> m_contextElement;
};

} // namespace WebCore
