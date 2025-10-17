/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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
#include "XPathExpressionNode.h"

#include <wtf/NeverDestroyed.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace XPath {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Expression);

EvaluationContext& Expression::evaluationContext()
{
    static NeverDestroyed<EvaluationContext> context;
    return context;
}

Expression::Expression()
    : m_isContextNodeSensitive(false)
    , m_isContextPositionSensitive(false)
    , m_isContextSizeSensitive(false)
{
}

void Expression::setSubexpressions(Vector<std::unique_ptr<Expression>> subexpressions)
{
    ASSERT(m_subexpressions.isEmpty());
    m_subexpressions = WTFMove(subexpressions);
    for (auto& subexpression : m_subexpressions) {
        m_isContextNodeSensitive |= subexpression->m_isContextNodeSensitive;
        m_isContextPositionSensitive |= subexpression->m_isContextPositionSensitive;
        m_isContextSizeSensitive |= subexpression->m_isContextSizeSensitive;
    }
}

} // namespace XPath
} // namespace WebCore
