/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 30, 2024.
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
#include "config.h"
#include "SVGTests.h"

#include "EventTarget.h"
#include "HTMLNames.h"
#include "NodeName.h"
#include "SVGElement.h"
#include "SVGNames.h"
#include "SVGStringList.h"
#include <wtf/Language.h>
#include <wtf/SortedArrayMap.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(MATHML)
#include "MathMLNames.h"
#endif

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGConditionalProcessingAttributes);

SVGConditionalProcessingAttributes::SVGConditionalProcessingAttributes(SVGElement& contextElement)
    : m_requiredExtensions(SVGStringList::create(&contextElement))
    , m_systemLanguage(SVGStringList::create(&contextElement))
{
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        SVGTests::PropertyRegistry::registerConditionalProcessingAttributeProperty<SVGNames::requiredExtensionsAttr, &SVGConditionalProcessingAttributes::m_requiredExtensions>();
        SVGTests::PropertyRegistry::registerConditionalProcessingAttributeProperty<SVGNames::systemLanguageAttr, &SVGConditionalProcessingAttributes::m_systemLanguage>();
    });
}

SVGTests::SVGTests(SVGElement* contextElement)
    : m_contextElement(*contextElement)
{
}

bool SVGTests::hasExtension(const String& extension)
{
    // We recognize XHTML and MathML, as implemented in Gecko and suggested in the SVG Tiny
    // recommendation (http://www.w3.org/TR/SVG11/struct.html#RequiredExtensionsAttribute).
#if ENABLE(MATHML)
    if (extension == MathMLNames::mathmlNamespaceURI)
        return true;
#endif
    return extension == HTMLNames::xhtmlNamespaceURI;
}

bool SVGTests::isValid() const
{
    auto attributes = conditionalProcessingAttributesIfExists();
    if (!attributes)
        return true;

    String defaultLanguage = WTF::defaultLanguage();
    auto genericDefaultLanguage = StringView(defaultLanguage).left(2);
    for (auto& language : attributes->systemLanguage().items()) {
        if (language != genericDefaultLanguage)
            return false;
    }
    for (auto& extension : attributes->requiredExtensions().items()) {
        if (!hasExtension(extension))
            return false;
    }
    return true;
}

Ref<SVGStringList> SVGTests::protectedRequiredExtensions()
{
    return requiredExtensions();
}

Ref<SVGStringList> SVGTests::protectedSystemLanguage()
{
    return systemLanguage();
}

void SVGTests::parseAttribute(const QualifiedName& attributeName, const AtomString& value)
{
    switch (attributeName.nodeName()) {
    case AttributeNames::requiredExtensionsAttr:
        protectedRequiredExtensions()->reset(value);
        break;
    case AttributeNames::systemLanguageAttr:
        protectedSystemLanguage()->reset(value);
        break;
    default:
        break;
    }
}

void SVGTests::svgAttributeChanged(const QualifiedName& attrName)
{
    if (!PropertyRegistry::isKnownAttribute(attrName))
        return;

    Ref contextElement = m_contextElement.get();
    if (!contextElement->isConnected())
        return;
    contextElement->invalidateStyleAndRenderersForSubtree();
}

void SVGTests::addSupportedAttributes(MemoryCompactLookupOnlyRobinHoodHashSet<QualifiedName>& supportedAttributes)
{
    supportedAttributes.add(SVGNames::requiredExtensionsAttr);
    supportedAttributes.add(SVGNames::systemLanguageAttr);
}

Ref<SVGElement> SVGTests::protectedContextElement() const
{
    return m_contextElement.get();
}

SVGConditionalProcessingAttributes& SVGTests::conditionalProcessingAttributes()
{
    Ref<SVGElement> contextElement = m_contextElement.get();
    return contextElement->conditionalProcessingAttributes();
}

SVGConditionalProcessingAttributes* SVGTests::conditionalProcessingAttributesIfExists() const
{
    return protectedContextElement()->conditionalProcessingAttributesIfExists();
}

} // namespace WebCore
