/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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
#include "XPathEvaluator.h"

#include "NativeXPathNSResolver.h"
#include "XPathExpression.h"
#include "XPathResult.h"
#include "XPathUtil.h"

namespace WebCore {

ExceptionOr<Ref<XPathExpression>> XPathEvaluator::createExpression(const String& expression, RefPtr<XPathNSResolver>&& resolver)
{
    return XPathExpression::createExpression(expression, WTFMove(resolver));
}

Ref<XPathNSResolver> XPathEvaluator::createNSResolver(Node& nodeResolver)
{
    return NativeXPathNSResolver::create(nodeResolver);
}

ExceptionOr<Ref<XPathResult>> XPathEvaluator::evaluate(const String& expression, Node& contextNode, RefPtr<XPathNSResolver>&& resolver, unsigned short type, XPathResult* result)
{
    if (!XPath::isValidContextNode(contextNode))
        return Exception { ExceptionCode::NotSupportedError };

    auto createResult = createExpression(expression, WTFMove(resolver));
    if (createResult.hasException())
        return createResult.releaseException();

    return createResult.releaseReturnValue()->evaluate(contextNode, type, result);
}

}
