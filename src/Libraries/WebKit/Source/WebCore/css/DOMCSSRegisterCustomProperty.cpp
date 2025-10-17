/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 30, 2024.
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
#include "DOMCSSRegisterCustomProperty.h"

#include "CSSCustomPropertyValue.h"
#include "CSSPropertyNames.h"
#include "CSSPropertyParser.h"
#include "CSSRegisteredCustomProperty.h"
#include "CSSTokenizer.h"
#include "CustomPropertyRegistry.h"
#include "DOMCSSNamespace.h"
#include "Document.h"
#include "StyleBuilder.h"
#include "StyleBuilderConverter.h"
#include "StyleResolver.h"
#include "StyleScope.h"
#include <wtf/text/WTFString.h>

namespace WebCore {
DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(DOMCSSRegisterCustomProperty);

using namespace Style;

ExceptionOr<void> DOMCSSRegisterCustomProperty::registerProperty(Document& document, const DOMCSSCustomPropertyDescriptor& descriptor)
{
    if (!isCustomPropertyName(descriptor.name))
        return Exception { ExceptionCode::SyntaxError, "The name of this property is not a custom property name."_s };

    auto syntax = CSSCustomPropertySyntax::parse(descriptor.syntax);
    if (!syntax)
        return Exception { ExceptionCode::SyntaxError, "Invalid property syntax definition."_s };

    if (!syntax->isUniversal() && descriptor.initialValue.isNull())
        return Exception { ExceptionCode::SyntaxError, "An initial value is mandatory except for the '*' syntax."_s };

    RefPtr<CSSCustomPropertyValue> initialValue;
    RefPtr<CSSVariableData> initialValueTokensForViewportUnits;

    if (!descriptor.initialValue.isNull()) {
        CSSTokenizer tokenizer(descriptor.initialValue);

        auto parsedInitialValue = CustomPropertyRegistry::parseInitialValue(document, descriptor.name, *syntax, tokenizer.tokenRange());

        if (!parsedInitialValue) {
            if (parsedInitialValue.error() == CustomPropertyRegistry::ParseInitialValueError::NotComputationallyIndependent)
                return Exception { ExceptionCode::SyntaxError, "The given initial value must be computationally independent."_s };

            ASSERT(parsedInitialValue.error() == CustomPropertyRegistry::ParseInitialValueError::DidNotParse);
            return Exception { ExceptionCode::SyntaxError, "The given initial value does not parse for the given syntax."_s };
        }

        initialValue = parsedInitialValue->first;
        if (parsedInitialValue->second == CustomPropertyRegistry::ViewportUnitDependency::Yes) {
            initialValueTokensForViewportUnits = CSSVariableData::create(tokenizer.tokenRange());
            document.setHasStyleWithViewportUnits();
        }
    }

    auto property = CSSRegisteredCustomProperty {
        descriptor.name,
        *syntax,
        descriptor.inherits,
        WTFMove(initialValue),
        WTFMove(initialValueTokensForViewportUnits)
    };

    auto& registry = document.styleScope().customPropertyRegistry();
    if (!registry.registerFromAPI(WTFMove(property)))
        return Exception { ExceptionCode::InvalidModificationError, "This property has already been registered."_s };

    return { };
}

DOMCSSRegisterCustomProperty* DOMCSSRegisterCustomProperty::from(DOMCSSNamespace& css)
{
    auto* supplement = static_cast<DOMCSSRegisterCustomProperty*>(Supplement<DOMCSSNamespace>::from(&css, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<DOMCSSRegisterCustomProperty>(css);
        supplement = newSupplement.get();
        provideTo(&css, supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

ASCIILiteral DOMCSSRegisterCustomProperty::supplementName()
{
    return "DOMCSSRegisterCustomProperty"_s;
}

}
