/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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

#include "CSSValueList.h"

namespace WebCore {

enum CSSValueID : uint16_t;

class CSSFunctionValue final : public CSSValueContainingVector {
public:
    static Ref<CSSFunctionValue> create(CSSValueID name, CSSValueListBuilder arguments);
    static Ref<CSSFunctionValue> create(CSSValueID name);
    static Ref<CSSFunctionValue> create(CSSValueID name, Ref<CSSValue> argument);
    static Ref<CSSFunctionValue> create(CSSValueID name, Ref<CSSValue> firstArgument, Ref<CSSValue> secondArgument);
    static Ref<CSSFunctionValue> create(CSSValueID name, Ref<CSSValue> firstArgument, Ref<CSSValue> secondArgument, Ref<CSSValue> thirdArgument);
    static Ref<CSSFunctionValue> create(CSSValueID name, Ref<CSSValue> firstArgument, Ref<CSSValue> secondArgument, Ref<CSSValue> thirdArgument, Ref<CSSValue> fourthArgument);

    CSSValueID name() const { return m_name; }

    String customCSSText() const;
    bool equals(const CSSFunctionValue& other) const { return m_name == other.m_name && itemsEqual(other); }

private:
    friend bool CSSValue::addHash(Hasher&) const;

    CSSFunctionValue(CSSValueID name, CSSValueListBuilder);
    explicit CSSFunctionValue(CSSValueID name);
    CSSFunctionValue(CSSValueID name, Ref<CSSValue>);
    CSSFunctionValue(CSSValueID name, Ref<CSSValue>, Ref<CSSValue>);
    CSSFunctionValue(CSSValueID name, Ref<CSSValue>, Ref<CSSValue>, Ref<CSSValue>);
    CSSFunctionValue(CSSValueID name, Ref<CSSValue>, Ref<CSSValue>, Ref<CSSValue>, Ref<CSSValue>);

    bool addDerivedHash(Hasher&) const;

    CSSValueID m_name { };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSFunctionValue, isFunctionValue())
