/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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
#include "XPathPredicate.h"

#include "XPathFunctions.h"
#include "XPathUtil.h"
#include <math.h>
#include <wtf/MathExtras.h>
#include <wtf/SetForScope.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace XPath {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Number);
WTF_MAKE_TZONE_ALLOCATED_IMPL(StringExpression);
WTF_MAKE_TZONE_ALLOCATED_IMPL(Negative);
WTF_MAKE_TZONE_ALLOCATED_IMPL(NumericOp);
WTF_MAKE_TZONE_ALLOCATED_IMPL(EqTestOp);
WTF_MAKE_TZONE_ALLOCATED_IMPL(LogicalOp);
WTF_MAKE_TZONE_ALLOCATED_IMPL(Union);

Number::Number(double value)
    : m_value(value)
{
}

Value Number::evaluate() const
{
    return m_value;
}

StringExpression::StringExpression(String&& value)
    : m_value(WTFMove(value))
{
}

Value StringExpression::evaluate() const
{
    return m_value;
}

Negative::Negative(std::unique_ptr<Expression> expression)
{
    addSubexpression(WTFMove(expression));
}

Value Negative::evaluate() const
{
    return -subexpression(0).evaluate().toNumber();
}

NumericOp::NumericOp(Opcode opcode, std::unique_ptr<Expression> lhs, std::unique_ptr<Expression> rhs)
    : m_opcode(opcode)
{
    addSubexpression(WTFMove(lhs));
    addSubexpression(WTFMove(rhs));
}

Value NumericOp::evaluate() const
{
    EvaluationContext clonedContext(Expression::evaluationContext());

    double leftVal = subexpression(0).evaluate().toNumber();
    double rightVal;

    {
        SetForScope contextForScope(Expression::evaluationContext(), clonedContext);
        rightVal = subexpression(1).evaluate().toNumber();
    }

    switch (m_opcode) {
    case Opcode::Add:
            return leftVal + rightVal;
    case Opcode::Sub:
            return leftVal - rightVal;
    case Opcode::Mul:
            return leftVal * rightVal;
    case Opcode::Div:
            return leftVal / rightVal;
    case Opcode::Mod:
            return fmod(leftVal, rightVal);
    }

    ASSERT_NOT_REACHED();
    return 0.0;
}

EqTestOp::EqTestOp(Opcode opcode, std::unique_ptr<Expression> lhs, std::unique_ptr<Expression> rhs)
    : m_opcode(opcode)
{
    addSubexpression(WTFMove(lhs));
    addSubexpression(WTFMove(rhs));
}

bool EqTestOp::compare(const Value& lhs, const Value& rhs) const
{
    if (lhs.isNodeSet()) {
        const NodeSet& lhsSet = lhs.toNodeSet();
        if (rhs.isNodeSet()) {
            // If both objects to be compared are node-sets, then the comparison will be true if and only if
            // there is a node in the first node-set and a node in the second node-set such that the result of
            // performing the comparison on the string-values of the two nodes is true.
            const NodeSet& rhsSet = rhs.toNodeSet();
            for (auto& lhs : lhsSet) {
                for (auto& rhs : rhsSet) {
                    if (compare(stringValue(lhs.get()), stringValue(rhs.get())))
                        return true;
                }
            }
            return false;
        }
        if (rhs.isNumber()) {
            // If one object to be compared is a node-set and the other is a number, then the comparison will be true
            // if and only if there is a node in the node-set such that the result of performing the comparison on the number
            // to be compared and on the result of converting the string-value of that node to a number using the number function is true.
            for (auto& lhs : lhsSet) {
                if (compare(Value(stringValue(lhs.get())).toNumber(), rhs))
                    return true;
            }
            return false;
        }
        if (rhs.isString()) {
            // If one object to be compared is a node-set and the other is a string, then the comparison will be true
            // if and only if there is a node in the node-set such that the result of performing the comparison on
            // the string-value of the node and the other string is true.
            for (auto& lhs : lhsSet) {
                if (compare(stringValue(lhs.get()), rhs))
                    return true;
            }
            return false;
        }
        if (rhs.isBoolean()) {
            // If one object to be compared is a node-set and the other is a boolean, then the comparison will be true
            // if and only if the result of performing the comparison on the boolean and on the result of converting
            // the node-set to a boolean using the boolean function is true.
            return compare(lhs.toBoolean(), rhs);
        }
        ASSERT_NOT_REACHED();
    }
    if (rhs.isNodeSet()) {
        const NodeSet& rhsSet = rhs.toNodeSet();
        if (lhs.isNumber()) {
            for (auto& rhs : rhsSet) {
                if (compare(lhs, Value(stringValue(rhs.get())).toNumber()))
                    return true;
            }
            return false;
        }
        if (lhs.isString()) {
            for (auto& rhs : rhsSet) {
                if (compare(lhs, stringValue(rhs.get())))
                    return true;
            }
            return false;
        }
        if (lhs.isBoolean())
            return compare(lhs, rhs.toBoolean());
        ASSERT_NOT_REACHED();
    }
    
    // Neither side is a NodeSet.
    switch (m_opcode) {
    case Opcode::Eq:
    case Opcode::Ne:
            bool equal;
            if (lhs.isBoolean() || rhs.isBoolean())
                equal = lhs.toBoolean() == rhs.toBoolean();
            else if (lhs.isNumber() || rhs.isNumber())
                equal = lhs.toNumber() == rhs.toNumber();
            else
                equal = lhs.toString() == rhs.toString();

            if (m_opcode == Opcode::Eq)
                return equal;
            return !equal;
    case Opcode::Gt:
            return lhs.toNumber() > rhs.toNumber();
    case Opcode::Ge:
            return lhs.toNumber() >= rhs.toNumber();
    case Opcode::Lt:
            return lhs.toNumber() < rhs.toNumber();
    case Opcode::Le:
            return lhs.toNumber() <= rhs.toNumber();
    }

    ASSERT_NOT_REACHED();
    return false;
}

Value EqTestOp::evaluate() const
{
    EvaluationContext clonedContext(Expression::evaluationContext());

    Value lhs(subexpression(0).evaluate());
    Value rhs = [&] {
        SetForScope contextForScope(Expression::evaluationContext(), clonedContext);
        return subexpression(1).evaluate();
    }();

    return compare(lhs, rhs);
}

LogicalOp::LogicalOp(Opcode opcode, std::unique_ptr<Expression> lhs, std::unique_ptr<Expression> rhs)
    : m_opcode(opcode)
{
    addSubexpression(WTFMove(lhs));
    addSubexpression(WTFMove(rhs));
}

inline bool LogicalOp::shortCircuitOn() const
{
    return m_opcode != Opcode::And;
}

Value LogicalOp::evaluate() const
{
    EvaluationContext clonedContext(Expression::evaluationContext());

    // This is not only an optimization, http://www.w3.org/TR/xpath
    // dictates that we must do short-circuit evaluation
    bool lhsBool = subexpression(0).evaluate().toBoolean();
    if (lhsBool == shortCircuitOn())
        return lhsBool;

    SetForScope contextForScope(Expression::evaluationContext(), clonedContext);
    return subexpression(1).evaluate().toBoolean();
}

Union::Union(std::unique_ptr<Expression> lhs, std::unique_ptr<Expression> rhs)
{
    addSubexpression(WTFMove(lhs));
    addSubexpression(WTFMove(rhs));
}

Value Union::evaluate() const
{
    EvaluationContext clonedContext(Expression::evaluationContext());
    Value lhsResult = subexpression(0).evaluate();
    Value rhsResult = [&] {
        SetForScope contextForScope(Expression::evaluationContext(), clonedContext);
        return subexpression(1).evaluate();
    }();
    Expression::evaluationContext().hadTypeConversionError |= clonedContext.hadTypeConversionError;

    NodeSet& resultSet = lhsResult.modifiableNodeSet();
    const NodeSet& rhsNodes = rhsResult.toNodeSet();

    HashSet<RefPtr<Node>> nodes;
    for (auto& result : resultSet)
        nodes.add(result.get());

    for (auto& node : rhsNodes) {
        if (nodes.add(node.get()).isNewEntry)
            resultSet.append(node.get());
    }

    // It would also be possible to perform a merge sort here to avoid making an unsorted result,
    // but that would waste the time in cases when order is not important.
    resultSet.markSorted(false);

    return lhsResult;
}

bool evaluatePredicate(const Expression& expression)
{
    EvaluationContext clonedContext(Expression::evaluationContext());
    Value result = [&] {
        SetForScope contextForScope(Expression::evaluationContext(), clonedContext);
        return expression.evaluate();
    }();
    Expression::evaluationContext().hadTypeConversionError |= clonedContext.hadTypeConversionError;

    // foo[3] means foo[position()=3]
    if (result.isNumber())
        return EqTestOp(EqTestOp::Opcode::Eq, Function::create("position"_s), makeUnique<Number>(result.toNumber())).evaluate().toBoolean();

    return result.toBoolean();
}

bool predicateIsContextPositionSensitive(const Expression& expression)
{
    return expression.isContextPositionSensitive() || expression.resultType() == Value::Type::Number;
}

} // namespace XPath
} // namespace WebCore
