/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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
#include "XPathValue.h"

#include "CommonAtomStrings.h"
#include "XPathExpressionNode.h"
#include "XPathUtil.h"
#include <limits>
#include <wtf/MathExtras.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/StdLibExtras.h>

namespace WebCore {
namespace XPath {

const NodeSet& Value::toNodeSet() const
{
    if (!isNodeSet())
        Expression::evaluationContext().hadTypeConversionError = true;

    if (!m_data) {
        static NeverDestroyed<NodeSet> emptyNodeSet;
        return emptyNodeSet;
    }

    return m_data->nodeSet;
}    

NodeSet& Value::modifiableNodeSet()
{
    if (!isNodeSet())
        Expression::evaluationContext().hadTypeConversionError = true;

    if (!m_data)
        m_data = Data::create();

    m_type = Type::NodeSet;
    return m_data->nodeSet;
}

bool Value::toBoolean() const
{
    switch (m_type) {
    case Type::NodeSet:
        return !m_data->nodeSet.isEmpty();
    case Type::Boolean:
        return m_bool;
    case Type::Number:
        return m_number && !std::isnan(m_number);
    case Type::String:
        return !m_data->string.isEmpty();
    }
    ASSERT_NOT_REACHED();
    return false;
}

double Value::toNumber() const
{
    switch (m_type) {
    case Type::NodeSet:
        return Value(toString()).toNumber();
    case Type::Number:
        return m_number;
    case Type::String: {
        const String& str = m_data->string.simplifyWhiteSpace(deprecatedIsSpaceOrNewline);

        // String::toDouble() supports exponential notation, which is not allowed in XPath.
        unsigned len = str.length();
        for (unsigned i = 0; i < len; ++i) {
            UChar c = str[i];
            if (!isASCIIDigit(c) && c != '.'  && c != '-')
                return std::numeric_limits<double>::quiet_NaN();
        }

        bool canConvert;
        double value = str.toDouble(&canConvert);
        if (canConvert)
            return value;
        return std::numeric_limits<double>::quiet_NaN();
    }
    case Type::Boolean:
        return m_bool;
    }

    ASSERT_NOT_REACHED();
    return 0.0;
}

String Value::toString() const
{
    switch (m_type) {
    case Type::NodeSet:
        if (m_data->nodeSet.isEmpty())
            return emptyString();
        return stringValue(m_data->nodeSet.firstNode());
    case Type::String:
        return m_data->string;
    case Type::Number:
        if (std::isnan(m_number))
            return "NaN"_s;
        if (!m_number)
            return "0"_s;
        if (std::isinf(m_number))
            return std::signbit(m_number) ? "-Infinity"_s : "Infinity"_s;
        return String::number(m_number);
    case Type::Boolean:
        return m_bool ? trueAtom() : falseAtom();
    }

    ASSERT_NOT_REACHED();
    return String();
}

} // namespace XPath
} // namespace WebCore
