/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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
#include "MainThreadStylePropertyMapReadOnly.h"

#include "CSSPendingSubstitutionValue.h"
#include "CSSProperty.h"
#include "CSSPropertyNames.h"
#include "CSSPropertyParser.h"
#include "CSSStyleValue.h"
#include "CSSStyleValueFactory.h"
#include "CSSTokenizer.h"
#include "CSSUnparsedValue.h"
#include "CSSVariableData.h"
#include "Document.h"
#include "PaintWorkletGlobalScope.h"
#include "StylePropertyShorthand.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

MainThreadStylePropertyMapReadOnly::MainThreadStylePropertyMapReadOnly() = default;

Document* MainThreadStylePropertyMapReadOnly::documentFromContext(ScriptExecutionContext& context)
{
    ASSERT(isMainThread());

    if (auto* paintWorklet = dynamicDowncast<PaintWorkletGlobalScope>(context))
        return paintWorklet->responsibleDocument();
    return &downcast<Document>(context);
}

// https://drafts.css-houdini.org/css-typed-om-1/#dom-stylepropertymapreadonly-get
ExceptionOr<MainThreadStylePropertyMapReadOnly::CSSStyleValueOrUndefined> MainThreadStylePropertyMapReadOnly::get(ScriptExecutionContext& context, const AtomString& property) const
{
    auto* document = documentFromContext(context);
    if (!document)
        return { std::monostate { } };

    if (isCustomPropertyName(property)) {
        if (auto value = reifyValue(customPropertyValue(property), std::nullopt, *document))
            return { WTFMove(value) };

        return { std::monostate { } };
    }

    auto propertyID = cssPropertyID(property);
    if (!isExposed(propertyID, &document->settings()))
        return Exception { ExceptionCode::TypeError, makeString("Invalid property "_s, property) };

    if (isShorthand(propertyID)) {
        if (auto value = CSSStyleValueFactory::constructStyleValueForShorthandSerialization(shorthandPropertySerialization(propertyID), { *document }))
            return { WTFMove(value) };

        return { std::monostate { } };
    }

    if (auto value = reifyValue(propertyValue(propertyID), propertyID, *document))
        return { WTFMove(value) };

    return { std::monostate { } };
}

// https://drafts.css-houdini.org/css-typed-om-1/#dom-stylepropertymapreadonly-getall
ExceptionOr<Vector<RefPtr<CSSStyleValue>>> MainThreadStylePropertyMapReadOnly::getAll(ScriptExecutionContext& context, const AtomString& property) const
{
    auto* document = documentFromContext(context);
    if (!document)
        return Vector<RefPtr<CSSStyleValue>> { };

    if (isCustomPropertyName(property))
        return reifyValueToVector(customPropertyValue(property), std::nullopt, *document);

    auto propertyID = cssPropertyID(property);
    if (!isExposed(propertyID, &document->settings()))
        return Exception { ExceptionCode::TypeError, makeString("Invalid property "_s, property) };

    if (isShorthand(propertyID)) {
        if (RefPtr value = CSSStyleValueFactory::constructStyleValueForShorthandSerialization(shorthandPropertySerialization(propertyID), { *document }))
            return Vector<RefPtr<CSSStyleValue>> { WTFMove(value) };
        return Vector<RefPtr<CSSStyleValue>> { };
    }

    return reifyValueToVector(propertyValue(propertyID), propertyID, *document);
}

// https://drafts.css-houdini.org/css-typed-om-1/#dom-stylepropertymapreadonly-has
ExceptionOr<bool> MainThreadStylePropertyMapReadOnly::has(ScriptExecutionContext& context, const AtomString& property) const
{
    auto result = get(context, property);
    if (result.hasException())
        return result.releaseException();

    return WTF::switchOn(result.returnValue(),
        [](const RefPtr<CSSStyleValue>& value) {
            ASSERT(value);
            return !!value;
        },
        [](std::monostate) {
            return false;
        }
    );
}

} // namespace WebCore
