/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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
#include "CSSValue.h"
#include <wtf/RefCounted.h>
#include <wtf/UniqueRef.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Document;
class Element;
class ScriptExecutionContext;
class StyledElement;

class StylePropertyMapReadOnly : public RefCounted<StylePropertyMapReadOnly> {
public:
    using StylePropertyMapEntry = KeyValuePair<String, Vector<RefPtr<CSSStyleValue>>>;

    enum class Type {
        Computed,
        Declared,
        UncheckedKeyHashMap,
        Inline,
    };

    virtual Type type() const = 0;

    class Iterator {
    public:
        explicit Iterator(StylePropertyMapReadOnly&, ScriptExecutionContext*);
        std::optional<StylePropertyMapEntry> next();

    private:
        Vector<StylePropertyMapEntry> m_values;
        size_t m_index { 0 };
    };
    Iterator createIterator(ScriptExecutionContext* context) { return Iterator(*this, context); }

    virtual ~StylePropertyMapReadOnly() = default;
    using CSSStyleValueOrUndefined = std::variant<std::monostate, RefPtr<CSSStyleValue>>;
    virtual ExceptionOr<CSSStyleValueOrUndefined> get(ScriptExecutionContext&, const AtomString& property) const = 0;
    virtual ExceptionOr<Vector<RefPtr<CSSStyleValue>>> getAll(ScriptExecutionContext&, const AtomString&) const = 0;
    virtual ExceptionOr<bool> has(ScriptExecutionContext&, const AtomString&) const = 0;
    virtual unsigned size() const = 0;

    static RefPtr<CSSStyleValue> reifyValue(RefPtr<CSSValue>&&, std::optional<CSSPropertyID>, Document&);
    static Vector<RefPtr<CSSStyleValue>> reifyValueToVector(RefPtr<CSSValue>&&, std::optional<CSSPropertyID>, Document&);

protected:
    virtual Vector<StylePropertyMapEntry> entries(ScriptExecutionContext*) const = 0;
};

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_CSSOM_STYLE_PROPERTY_MAP(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ToValueTypeName) \
    static bool isType(const WebCore::StylePropertyMapReadOnly& value) { return value.type() == predicate; } \
SPECIALIZE_TYPE_TRAITS_END()
