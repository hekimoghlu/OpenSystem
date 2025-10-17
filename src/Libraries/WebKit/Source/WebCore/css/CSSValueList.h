/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 20, 2022.
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

#include "CSSValue.h"
#include <array>
#include <unicode/umachine.h>
#include <wtf/MallocSpan.h>

namespace WebCore {

static constexpr size_t CSSValueListBuilderInlineCapacity = 4;
using CSSValueListBuilder = Vector<Ref<CSSValue>, CSSValueListBuilderInlineCapacity>;

class CSSValueContainingVector : public CSSValue {
public:
    unsigned size() const { return m_size; }
    const CSSValue& operator[](unsigned index) const;

    struct iterator {
        using iterator_category = std::forward_iterator_tag;
        using value_type = const CSSValue&;
        using difference_type = ptrdiff_t;
        using pointer = const CSSValue*;
        using reference = const CSSValue&;

        const CSSValue& operator*() const { return vector[index]; }
        iterator& operator++() { ++index; return *this; }
        constexpr bool operator==(const iterator& other) const { return index == other.index; }

        const CSSValueContainingVector& vector;
        unsigned index { 0 };
    };
    using const_iterator = iterator;

    iterator begin() const { return { *this, 0 }; }
    iterator end() const { return { *this, size() }; }

    bool hasValue(CSSValue&) const;
    bool hasValue(CSSValueID) const;

    void serializeItems(StringBuilder&) const;
    String serializeItems() const;

    bool itemsEqual(const CSSValueContainingVector&) const;
    bool containsSingleEqualItem(const CSSValue&) const;

    using CSSValue::separator;
    using CSSValue::separatorCSSText;

    bool customTraverseSubresources(const Function<bool(const CachedResource&)>&) const;
    void customSetReplacementURLForSubresources(const UncheckedKeyHashMap<String, String>&);
    void customClearReplacementURLForSubresources();

    CSSValueListBuilder copyValues() const;

    // Consider removing these functions and having callers use size() and operator[] instead.
    unsigned length() const { return size(); }
    const CSSValue* item(unsigned index) const { return index < size() ? &(*this)[index] : nullptr; }
    RefPtr<const CSSValue> protectedItem(unsigned index) const { return item(index); }
    const CSSValue* itemWithoutBoundsCheck(unsigned index) const { return &(*this)[index]; }

    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>&) const;

protected:
    friend bool CSSValue::addHash(Hasher&) const;

    CSSValueContainingVector(ClassType, ValueSeparator);
    CSSValueContainingVector(ClassType, ValueSeparator, CSSValueListBuilder);
    CSSValueContainingVector(ClassType, ValueSeparator, Ref<CSSValue>);
    CSSValueContainingVector(ClassType, ValueSeparator, Ref<CSSValue>, Ref<CSSValue>);
    CSSValueContainingVector(ClassType, ValueSeparator, Ref<CSSValue>, Ref<CSSValue>, Ref<CSSValue>);
    CSSValueContainingVector(ClassType, ValueSeparator, Ref<CSSValue>, Ref<CSSValue>, Ref<CSSValue>, Ref<CSSValue>);
    ~CSSValueContainingVector();

    bool addDerivedHash(Hasher&) const;

private:
    unsigned m_size { 0 };
    std::array<const CSSValue*, 4> m_inlineStorage;
    MallocSpan<const CSSValue*> m_additionalStorage;
};

class CSSValueList final : public CSSValueContainingVector {
public:
    static Ref<CSSValueList> create(UChar separator, CSSValueListBuilder);

    static Ref<CSSValueList> createCommaSeparated(CSSValueListBuilder);
    static Ref<CSSValueList> createCommaSeparated(Ref<CSSValue>); // FIXME: Upgrade callers to not use a list at all.

    static Ref<CSSValueList> createSpaceSeparated(CSSValueListBuilder);
    static Ref<CSSValueList> createSpaceSeparated(); // FIXME: Get rid of the caller that needs to create an empty list.
    static Ref<CSSValueList> createSpaceSeparated(Ref<CSSValue>); // FIXME: Upgrade callers to not use a list at all.
    static Ref<CSSValueList> createSpaceSeparated(Ref<CSSValue>, Ref<CSSValue>);
    static Ref<CSSValueList> createSpaceSeparated(Ref<CSSValue>, Ref<CSSValue>, Ref<CSSValue>);
    static Ref<CSSValueList> createSpaceSeparated(Ref<CSSValue>, Ref<CSSValue>, Ref<CSSValue>, Ref<CSSValue>);

    static Ref<CSSValueList> createSlashSeparated(CSSValueListBuilder);
    static Ref<CSSValueList> createSlashSeparated(Ref<CSSValue>); // FIXME: Upgrade callers to not use a list at all.
    static Ref<CSSValueList> createSlashSeparated(Ref<CSSValue>, Ref<CSSValue>);

    String customCSSText() const;
    bool equals(const CSSValueList&) const;

private:
    friend void add(Hasher&, const CSSValueList&);

    explicit CSSValueList(ValueSeparator);
    CSSValueList(ValueSeparator, CSSValueListBuilder);
    CSSValueList(ValueSeparator, Ref<CSSValue>);
    CSSValueList(ValueSeparator, Ref<CSSValue>, Ref<CSSValue>);
    CSSValueList(ValueSeparator, Ref<CSSValue>, Ref<CSSValue>, Ref<CSSValue>);
    CSSValueList(ValueSeparator, Ref<CSSValue>, Ref<CSSValue>, Ref<CSSValue>, Ref<CSSValue>);
};

inline CSSValueContainingVector::~CSSValueContainingVector()
{
    for (auto& value : *this)
        value.deref();
}

inline const CSSValue& CSSValueContainingVector::operator[](unsigned index) const
{
    unsigned maxInlineSize = m_inlineStorage.size();
    if (index < maxInlineSize) {
        ASSERT(index < m_size);
        return *m_inlineStorage[index];
    }
    ASSERT(index < m_size);
    return *m_additionalStorage[index - maxInlineSize];
}

void add(Hasher&, const CSSValueContainingVector&);

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSValueContainingVector, containsVector())
SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSValueList, isValueList())
