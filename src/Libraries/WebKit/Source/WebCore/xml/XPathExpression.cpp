/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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
#include "XPathExpression.h"

#include "Document.h"
#include "XPathNSResolver.h"
#include "XPathParser.h"
#include "XPathResult.h"
#include "XPathUtil.h"

namespace WebCore {
    
inline XPathExpression::XPathExpression(std::unique_ptr<XPath::Expression> expression)
    : m_topExpression(WTFMove(expression))
{
}

ExceptionOr<Ref<XPathExpression>> XPathExpression::createExpression(const String& expression, RefPtr<XPathNSResolver>&& resolver)
{
    auto parseResult = XPath::Parser::parseStatement(expression, WTFMove(resolver));
    if (parseResult.hasException())
        return parseResult.releaseException();

    return adoptRef(*new XPathExpression(parseResult.releaseReturnValue()));
}

XPathExpression::~XPathExpression() = default;

// FIXME: Why does this take an XPathResult that it ignores?
ExceptionOr<Ref<XPathResult>> XPathExpression::evaluate(Node& contextNode, unsigned short type, XPathResult*)
{
    if (!XPath::isValidContextNode(contextNode))
        return Exception { ExceptionCode::NotSupportedError };

    auto& evaluationContext = XPath::Expression::evaluationContext();
    evaluationContext.node = &contextNode;
    evaluationContext.size = 1;
    evaluationContext.position = 1;
    evaluationContext.hadTypeConversionError = false;
    auto result = XPathResult::create(contextNode.document(), m_topExpression->evaluate());
    evaluationContext.node = nullptr; // Do not hold a reference to the context node, as this may prevent the whole document from being destroyed in time.

    if (evaluationContext.hadTypeConversionError)
        return Exception { ExceptionCode::SyntaxError };

    if (type != XPathResult::ANY_TYPE) {
        auto convertToResult = result->convertTo(type);
        if (convertToResult.hasException())
            return convertToResult.releaseException();
    }

    return result;
}

}
