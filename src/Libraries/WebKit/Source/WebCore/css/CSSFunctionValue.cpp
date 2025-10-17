/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 9, 2023.
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
#include "CSSFunctionValue.h"

#include "CSSValueKeywords.h"
#include <wtf/Hasher.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
    
CSSFunctionValue::CSSFunctionValue(CSSValueID name, CSSValueListBuilder arguments)
    : CSSValueContainingVector(ClassType::Function, CommaSeparator, WTFMove(arguments))
    , m_name(name)
{
}

CSSFunctionValue::CSSFunctionValue(CSSValueID name)
    : CSSValueContainingVector(ClassType::Function, CommaSeparator)
    , m_name(name)
{
}

CSSFunctionValue::CSSFunctionValue(CSSValueID name, Ref<CSSValue> argument)
    : CSSValueContainingVector(ClassType::Function, CommaSeparator, WTFMove(argument))
    , m_name(name)
{
}

CSSFunctionValue::CSSFunctionValue(CSSValueID name, Ref<CSSValue> argument1, Ref<CSSValue> argument2)
    : CSSValueContainingVector(ClassType::Function, CommaSeparator, WTFMove(argument1), WTFMove(argument2))
    , m_name(name)
{
}

CSSFunctionValue::CSSFunctionValue(CSSValueID name, Ref<CSSValue> argument1, Ref<CSSValue> argument2, Ref<CSSValue> argument3)
    : CSSValueContainingVector(ClassType::Function, CommaSeparator, WTFMove(argument1), WTFMove(argument2), WTFMove(argument3))
    , m_name(name)
{
}

CSSFunctionValue::CSSFunctionValue(CSSValueID name, Ref<CSSValue> argument1, Ref<CSSValue> argument2, Ref<CSSValue> argument3, Ref<CSSValue> argument4)
    : CSSValueContainingVector(ClassType::Function, CommaSeparator, WTFMove(argument1), WTFMove(argument2), WTFMove(argument3), WTFMove(argument4))
    , m_name(name)
{
}

Ref<CSSFunctionValue> CSSFunctionValue::create(CSSValueID name, CSSValueListBuilder arguments)
{
    return adoptRef(*new CSSFunctionValue(name, WTFMove(arguments)));
}

Ref<CSSFunctionValue> CSSFunctionValue::create(CSSValueID name)
{
    return adoptRef(*new CSSFunctionValue(name));
}

Ref<CSSFunctionValue> CSSFunctionValue::create(CSSValueID name, Ref<CSSValue> argument)
{
    return adoptRef(*new CSSFunctionValue(name, WTFMove(argument)));
}

Ref<CSSFunctionValue> CSSFunctionValue::create(CSSValueID name, Ref<CSSValue> argument1, Ref<CSSValue> argument2)
{
    return adoptRef(*new CSSFunctionValue(name, WTFMove(argument1), WTFMove(argument2)));
}

Ref<CSSFunctionValue> CSSFunctionValue::create(CSSValueID name, Ref<CSSValue> argument1, Ref<CSSValue> argument2, Ref<CSSValue> argument3)
{
    return adoptRef(*new CSSFunctionValue(name, WTFMove(argument1), WTFMove(argument2), WTFMove(argument3)));
}

Ref<CSSFunctionValue> CSSFunctionValue::create(CSSValueID name, Ref<CSSValue> argument1, Ref<CSSValue> argument2, Ref<CSSValue> argument3, Ref<CSSValue> argument4)
{
    return adoptRef(*new CSSFunctionValue(name, WTFMove(argument1), WTFMove(argument2), WTFMove(argument3), WTFMove(argument4)));
}

String CSSFunctionValue::customCSSText() const
{
    StringBuilder result;
    result.append(nameLiteral(m_name), '(');
    serializeItems(result);
    result.append(')');
    return result.toString();
}

bool CSSFunctionValue::addDerivedHash(Hasher& hasher) const
{
    add(hasher, m_name);
    return CSSValueContainingVector::addDerivedHash(hasher);
}

}
