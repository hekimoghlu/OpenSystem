/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 29, 2024.
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
#include <wtf/Function.h>
#include <wtf/HashMap.h>
#include <wtf/IterationStatus.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

class StyleCustomPropertyData : public RefCounted<StyleCustomPropertyData> {
private:
    using CustomPropertyValueMap = UncheckedKeyHashMap<AtomString, RefPtr<const CSSCustomPropertyValue>>;

public:
    static Ref<StyleCustomPropertyData> create() { return adoptRef(*new StyleCustomPropertyData); }
    Ref<StyleCustomPropertyData> copy() const { return adoptRef(*new StyleCustomPropertyData(*this)); }

    bool operator==(const StyleCustomPropertyData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleCustomPropertyData&) const;
#endif

    const CSSCustomPropertyValue* get(const AtomString&) const;
    void set(const AtomString&, Ref<const CSSCustomPropertyValue>&&);

    unsigned size() const { return m_size; }
    bool mayHaveAnimatableProperties() const { return m_mayHaveAnimatableProperties; }

    void forEach(const Function<IterationStatus(const KeyValuePair<AtomString, RefPtr<const CSSCustomPropertyValue>>&)>&) const;
    AtomString findKeyAtIndex(unsigned) const;

private:
    StyleCustomPropertyData() = default;
    StyleCustomPropertyData(const StyleCustomPropertyData&);

    template<typename Callback> void forEachInternal(Callback&&) const;

    RefPtr<const StyleCustomPropertyData> m_parentValues;
    CustomPropertyValueMap m_ownValues;
    unsigned m_size { 0 };
    unsigned m_ancestorCount { 0 };
    bool m_mayHaveAnimatableProperties { false };
#if ASSERT_ENABLED
    mutable bool m_hasChildren { false };
#endif
};

} // namespace WebCore
