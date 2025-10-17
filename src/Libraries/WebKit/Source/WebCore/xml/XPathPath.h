/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 30, 2023.
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

class Step;

class Filter final : public Expression {
    WTF_MAKE_TZONE_ALLOCATED(Filter);
public:
    Filter(std::unique_ptr<Expression>, Vector<std::unique_ptr<Expression>> predicates);

private:
    Value evaluate() const override;
    Value::Type resultType() const override { return Value::Type::NodeSet; }

    std::unique_ptr<Expression> m_expression;
    Vector<std::unique_ptr<Expression>> m_predicates;
};

class LocationPath final : public Expression {
    WTF_MAKE_TZONE_ALLOCATED(LocationPath);
public:
    LocationPath();

    void setAbsolute() { m_isAbsolute = true; setIsContextNodeSensitive(false); }

    void evaluate(NodeSet& nodes) const; // nodes is an input/output parameter

    void appendStep(std::unique_ptr<Step>);
    void prependStep(std::unique_ptr<Step>);

private:
    Value evaluate() const override;
    Value::Type resultType() const override { return Value::Type::NodeSet; }

    Vector<std::unique_ptr<Step>> m_steps;
    bool m_isAbsolute;
};

class Path final : public Expression {
    WTF_MAKE_TZONE_ALLOCATED(Path);
public:
    Path(std::unique_ptr<Expression> filter, std::unique_ptr<LocationPath>);

private:
    Value evaluate() const override;
    Value::Type resultType() const override { return Value::Type::NodeSet; }

    std::unique_ptr<Expression> m_filter;
    std::unique_ptr<LocationPath> m_path;
};

} // namespace XPath
} // namespace WebCore
