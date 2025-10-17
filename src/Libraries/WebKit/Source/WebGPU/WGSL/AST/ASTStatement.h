/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 5, 2022.
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

#include "ASTBuilder.h"
#include "ASTNode.h"
#include <wtf/ReferenceWrapperVector.h>

namespace WGSL::AST {

class Statement : public Node {
    WGSL_AST_BUILDER_NODE(Statement);
public:
    using Ref = std::reference_wrapper<Statement>;
    using Ptr = Statement*;
    using List = ReferenceWrapperVector<Statement>;

protected:
    Statement(SourceSpan span)
        : Node(span)
    { }
};

} // namespace WGSL::AST

SPECIALIZE_TYPE_TRAITS_BEGIN(WGSL::AST::Statement)
static bool isType(const WGSL::AST::Node& node)
{
    switch (node.kind()) {
    case WGSL::AST::NodeKind::AssignmentStatement:
    case WGSL::AST::NodeKind::BreakStatement:
    case WGSL::AST::NodeKind::CallStatement:
    case WGSL::AST::NodeKind::CompoundAssignmentStatement:
    case WGSL::AST::NodeKind::CompoundStatement:
    case WGSL::AST::NodeKind::ContinueStatement:
    case WGSL::AST::NodeKind::DecrementIncrementStatement:
    case WGSL::AST::NodeKind::DiscardStatement:
    case WGSL::AST::NodeKind::ForStatement:
    case WGSL::AST::NodeKind::IfStatement:
    case WGSL::AST::NodeKind::LoopStatement:
    case WGSL::AST::NodeKind::PhonyAssignmentStatement:
    case WGSL::AST::NodeKind::ReturnStatement:
    case WGSL::AST::NodeKind::StaticAssertStatement:
    case WGSL::AST::NodeKind::SwitchStatement:
    case WGSL::AST::NodeKind::VariableStatement:
    case WGSL::AST::NodeKind::WhileStatement:
        return true;
    default:
        return false;
    }
}
SPECIALIZE_TYPE_TRAITS_END()
