/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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

#include "ASTForward.h"
#include "ASTInterpolateAttribute.h"
#include "CompilationMessage.h"

#include <wtf/Expected.h>

namespace WGSL {

class ShaderModule;

namespace AST {

class Visitor {
public:
    virtual ~Visitor() = default;

    // Shader Module
    virtual void visit(ShaderModule&);

    // Directive
    virtual void visit(AST::Directive&);
    virtual void visit(AST::DiagnosticDirective&);

    // Declaration
    virtual void visit(AST::Declaration&);
    virtual void visit(AST::Function&);
    virtual void visit(AST::Variable&);
    virtual void visit(AST::Structure&);
    virtual void visit(AST::TypeAlias&);
    virtual void visit(AST::ConstAssert&);

    // Attribute
    virtual void visit(AST::Attribute&);
    virtual void visit(AST::AlignAttribute&);
    virtual void visit(AST::BindingAttribute&);
    virtual void visit(AST::BuiltinAttribute&);
    virtual void visit(AST::ConstAttribute&);
    virtual void visit(AST::DiagnosticAttribute&);
    virtual void visit(AST::GroupAttribute&);
    virtual void visit(AST::IdAttribute&);
    virtual void visit(AST::InterpolateAttribute&);
    virtual void visit(AST::InvariantAttribute&);
    virtual void visit(AST::LocationAttribute&);
    virtual void visit(AST::MustUseAttribute&);
    virtual void visit(AST::SizeAttribute&);
    virtual void visit(AST::StageAttribute&);
    virtual void visit(AST::WorkgroupSizeAttribute&);

    // Expression
    virtual void visit(AST::Expression&);
    virtual void visit(AST::AbstractFloatLiteral&);
    virtual void visit(AST::AbstractIntegerLiteral&);
    virtual void visit(AST::BinaryExpression&);
    virtual void visit(AST::BitcastExpression&);
    virtual void visit(AST::BoolLiteral&);
    virtual void visit(AST::CallExpression&);
    virtual void visit(AST::FieldAccessExpression&);
    virtual void visit(AST::Float32Literal&);
    virtual void visit(AST::Float16Literal&);
    virtual void visit(AST::IdentifierExpression&);
    virtual void visit(AST::IdentityExpression&);
    virtual void visit(AST::IndexAccessExpression&);
    virtual void visit(AST::PointerDereferenceExpression&);
    virtual void visit(AST::Signed32Literal&);
    virtual void visit(AST::UnaryExpression&);
    virtual void visit(AST::Unsigned32Literal&);

    virtual void visit(AST::Parameter&);

    virtual void visit(AST::Identifier&);

    // Statement
    virtual void visit(AST::Statement&);
    virtual void visit(AST::AssignmentStatement&);
    virtual void visit(AST::BreakStatement&);
    virtual void visit(AST::CallStatement&);
    virtual void visit(AST::CompoundAssignmentStatement&);
    virtual void visit(AST::CompoundStatement&);
    virtual void visit(AST::ConstAssertStatement&);
    virtual void visit(AST::ContinueStatement&);
    virtual void visit(AST::DecrementIncrementStatement&);
    virtual void visit(AST::DiscardStatement&);
    virtual void visit(AST::ForStatement&);
    virtual void visit(AST::IfStatement&);
    virtual void visit(AST::LoopStatement&);
    virtual void visit(AST::PhonyAssignmentStatement&);
    virtual void visit(AST::ReturnStatement&);
    virtual void visit(AST::StaticAssertStatement&);
    virtual void visit(AST::SwitchStatement&);
    virtual void visit(AST::VariableStatement&);
    virtual void visit(AST::WhileStatement&);

    virtual void visit(AST::ArrayTypeExpression&);
    virtual void visit(AST::ElaboratedTypeExpression&);
    virtual void visit(AST::ReferenceTypeExpression&);

    virtual void visit(AST::StructureMember&);
    virtual void visit(AST::VariableQualifier&);
    virtual void visit(AST::SwitchClause&);
    virtual void visit(AST::Continuing&);

    bool hasError() const;
    Result<void> result();

    template<typename T> void checkErrorAndVisit(T& x)
    {
        if (!hasError())
            visit(x);
    }

    template<typename T> void maybeCheckErrorAndVisit(T* x)
    {
        if (!hasError() && x)
            visit(*x);
    }

protected:
    void setError(Error error)
    {
        ASSERT(!hasError());
        m_expectedError = makeUnexpected(error);
    }

private:
    Result<void> m_expectedError;
};

} // namespace AST
} // namespace WGSL
