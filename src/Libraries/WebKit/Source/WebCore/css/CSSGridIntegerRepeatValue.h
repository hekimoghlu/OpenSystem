/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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

#include "CSSPrimitiveValue.h"
#include "CSSValueList.h"

namespace WebCore {

// CSSGridIntegerRepeatValue stores the track sizes and line numbers when the
// integer-repeat syntax is used.
//
// Right now the integer-repeat syntax is as follows:
// <track-repeat> = repeat( [ <positive-integer> ],
//                          [ <line-names>? <track-size> ]+ <line-names>? )
// <fixed-repeat> = repeat( [ <positive-integer> ],
//                          [ <line-names>? <fixed-size> ]+ <line-names>? )
class CSSGridIntegerRepeatValue final : public CSSValueContainingVector {
public:
    static Ref<CSSGridIntegerRepeatValue> create(Ref<CSSPrimitiveValue>&& repetitions, CSSValueListBuilder);

    const CSSPrimitiveValue& repetitions() const { return m_repetitions; }

    String customCSSText() const;
    bool equals(const CSSGridIntegerRepeatValue&) const;

    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
    {
        if (CSSValueContainingVector::customVisitChildren(func) == IterationStatus::Done)
            return IterationStatus::Done;
        if (func(m_repetitions.get()) == IterationStatus::Done)
            return IterationStatus::Done;
        return IterationStatus::Continue;
    }

private:
    CSSGridIntegerRepeatValue(Ref<CSSPrimitiveValue>&& repetitions, CSSValueListBuilder);

    Ref<CSSPrimitiveValue> m_repetitions;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSGridIntegerRepeatValue, isGridIntegerRepeatValue());
