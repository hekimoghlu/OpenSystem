/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 13, 2022.
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

#include "CSSCustomPropertyValue.h"
#include "CSSImageValue.h"
#include "CSSPropertyNames.h"
#include "CSSStyleValue.h"
#include "CSSValue.h"
#include "ScriptWrappable.h"
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

template<typename T> class ExceptionOr;
struct CSSParserContext;
class CSSUnparsedValue;
class Document;
class StylePropertyShorthand;

class CSSStyleValueFactory {
public:
    static ExceptionOr<Ref<CSSStyleValue>> reifyValue(const CSSValue&, std::optional<CSSPropertyID>, Document* = nullptr);
    static ExceptionOr<Vector<Ref<CSSStyleValue>>> parseStyleValue(const AtomString&, const String&, bool parseMultiple, const CSSParserContext&);
    static RefPtr<CSSStyleValue> constructStyleValueForShorthandSerialization(const String&, const CSSParserContext&);
    static ExceptionOr<Vector<Ref<CSSStyleValue>>> vectorFromStyleValuesOrStrings(const AtomString& property, FixedVector<std::variant<RefPtr<CSSStyleValue>, String>>&& values, const CSSParserContext&);

    static RefPtr<CSSStyleValue> constructStyleValueForCustomPropertySyntaxValue(const CSSCustomPropertyValue::SyntaxValue&);

protected:
    CSSStyleValueFactory() = delete;

private:
    static ExceptionOr<RefPtr<CSSValue>> extractCSSValue(const CSSPropertyID&, const String&, const CSSParserContext&);
    static ExceptionOr<RefPtr<CSSStyleValue>> extractShorthandCSSValues(const CSSPropertyID&, const String&, const CSSParserContext&);
    static ExceptionOr<Ref<CSSUnparsedValue>> extractCustomCSSValues(const String&);
};

} // namespace WebCore
