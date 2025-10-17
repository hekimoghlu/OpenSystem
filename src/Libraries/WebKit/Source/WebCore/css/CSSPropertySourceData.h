/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 4, 2025.
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

#include "StyleRuleType.h"
#include <utility>
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class StyleRuleBase;

struct SourceRange {
    SourceRange();
    SourceRange(unsigned start, unsigned end);
    unsigned length() const;

    unsigned start;
    unsigned end;
};

struct CSSPropertySourceData {
    CSSPropertySourceData(const String& name, const String& value, bool important, bool disabled, bool parsedOk, const SourceRange&);
    CSSPropertySourceData(const CSSPropertySourceData& other);
    CSSPropertySourceData();

    String toString() const;
    unsigned hash() const;

    String name;
    String value;
    bool important { false };
    bool disabled { false };
    bool parsedOk { false };
    SourceRange range;
};

struct CSSStyleSourceData : public RefCounted<CSSStyleSourceData> {
    static Ref<CSSStyleSourceData> create()
    {
        return adoptRef(*new CSSStyleSourceData);
    }

    Vector<CSSPropertySourceData> propertyData;
};

struct CSSRuleSourceData;
typedef Vector<Ref<CSSRuleSourceData>> RuleSourceDataList;
typedef Vector<SourceRange> SelectorRangeList;

struct CSSRuleSourceData : public RefCounted<CSSRuleSourceData> {
    static Ref<CSSRuleSourceData> create(StyleRuleType type)
    {
        return adoptRef(*new CSSRuleSourceData(type));
    }

    CSSRuleSourceData(StyleRuleType type)
        : type(type)
        , styleSourceData(CSSStyleSourceData::create())
    {
    }

    StyleRuleType type;

    // Range of the selector list in the enclosing source.
    SourceRange ruleHeaderRange;

    // Range of the rule body (e.g. style text for style rules) in the enclosing source.
    SourceRange ruleBodyRange;

    // Only for CSSStyleRules. Not applicable if `isImplicitlyNested`.
    SelectorRangeList selectorRanges;

    RefPtr<CSSStyleSourceData> styleSourceData;

    RuleSourceDataList childRules;

    bool isImplicitlyNested { false };
    bool containsImplicitlyNestedProperties { false };
};

} // namespace WebCore
