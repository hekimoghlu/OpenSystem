/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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

#include "XPathExpressionNode.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {
namespace XPath {

class Number final : public Expression {
    WTF_MAKE_TZONE_ALLOCATED(Number);
public:
    explicit Number(double);

private:
    Value evaluate() const override;
    Value::Type resultType() const override { return Value::Type::Number; }

    Value m_value;
};

class StringExpression final : public Expression {
    WTF_MAKE_TZONE_ALLOCATED(StringExpression);
public:
    explicit StringExpression(String&&);

private:
    Value evaluate() const override;
    Value::Type resultType() const override { return Value::Type::String; }

    Value m_value;
};

class Negative final : public Expression {
    WTF_MAKE_TZONE_ALLOCATED(Negative);
public:
    explicit Negative(std::unique_ptr<Expression>);

private:
    Value evaluate() const override;
    Value::Type resultType() const override { return Value::Type::Number; }
};

class NumericOp final : public Expression {
    WTF_MAKE_TZONE_ALLOCATED(NumericOp);
public:
    enum class Opcode : uint8_t { Add, Sub, Mul, Div, Mod };
    NumericOp(Opcode, std::unique_ptr<Expression> lhs, std::unique_ptr<Expression> rhs);

private:
    Value evaluate() const override;
    Value::Type resultType() const override { return Value::Type::Number; }

    Opcode m_opcode;
};

class EqTestOp final : public Expression {
    WTF_MAKE_TZONE_ALLOCATED(EqTestOp);
public:
    enum class Opcode : uint8_t { Eq, Ne, Gt, Lt, Ge, Le };
    EqTestOp(Opcode, std::unique_ptr<Expression> lhs, std::unique_ptr<Expression> rhs);
    Value evaluate() const override;

private:
    Value::Type resultType() const override { return Value::Type::Boolean; }
    bool compare(const Value&, const Value&) const;

    Opcode m_opcode;
};

class LogicalOp final : public Expression {
    WTF_MAKE_TZONE_ALLOCATED(LogicalOp);
public:
    enum class Opcode : bool { And, Or };
    LogicalOp(Opcode, std::unique_ptr<Expression> lhs, std::unique_ptr<Expression> rhs);

private:
    Value::Type resultType() const override { return Value::Type::Boolean; }
    bool shortCircuitOn() const;
    Value evaluate() const override;

    Opcode m_opcode;
};

class Union final : public Expression {
    WTF_MAKE_TZONE_ALLOCATED(Union);
public:
    Union(std::unique_ptr<Expression>, std::unique_ptr<Expression>);

private:
    Value evaluate() const override;
    Value::Type resultType() const override { return Value::Type::NodeSet; }
};

bool evaluatePredicate(const Expression&);
bool predicateIsContextPositionSensitive(const Expression&);

} // namespace XPath
} // namespace WebCore
