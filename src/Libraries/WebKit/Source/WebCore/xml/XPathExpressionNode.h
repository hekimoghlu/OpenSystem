/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 17, 2024.
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

#include "XPathValue.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {
namespace XPath {

struct EvaluationContext {
    RefPtr<Node> node;
    unsigned size;
    unsigned position;
    HashMap<String, String> variableBindings;

    bool hadTypeConversionError;
};

class Expression {
    WTF_MAKE_TZONE_ALLOCATED(Expression);
    WTF_MAKE_NONCOPYABLE(Expression);
public:
    static EvaluationContext& evaluationContext();

    virtual ~Expression() = default;

    virtual Value evaluate() const = 0;
    virtual Value::Type resultType() const = 0;

    bool isContextNodeSensitive() const { return m_isContextNodeSensitive; }
    bool isContextPositionSensitive() const { return m_isContextPositionSensitive; }
    bool isContextSizeSensitive() const { return m_isContextSizeSensitive; }

protected:
    Expression();

    unsigned subexpressionCount() const { return m_subexpressions.size(); }
    const Expression& subexpression(unsigned i) const { return *m_subexpressions[i]; }

    void addSubexpression(std::unique_ptr<Expression> expression)
    {
        m_isContextNodeSensitive |= expression->m_isContextNodeSensitive;
        m_isContextPositionSensitive |= expression->m_isContextPositionSensitive;
        m_isContextSizeSensitive |= expression->m_isContextSizeSensitive;
        m_subexpressions.append(WTFMove(expression));
    }

    void setSubexpressions(Vector<std::unique_ptr<Expression>>);

    void setIsContextNodeSensitive(bool value) { m_isContextNodeSensitive = value; }
    void setIsContextPositionSensitive(bool value) { m_isContextPositionSensitive = value; }
    void setIsContextSizeSensitive(bool value) { m_isContextSizeSensitive = value; }

private:
    Vector<std::unique_ptr<Expression>> m_subexpressions;

    // Evaluation details that can be used for optimization.
    bool m_isContextNodeSensitive;
    bool m_isContextPositionSensitive;
    bool m_isContextSizeSensitive;
};

} // namespace XPath
} // namespace WebCore
