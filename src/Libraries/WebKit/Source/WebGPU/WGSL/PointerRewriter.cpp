/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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
#include "PointerRewriter.h"

#include "AST.h"
#include "ASTScopedVisitorInlines.h"
#include "CallGraph.h"
#include "ContextProviderInlines.h"
#include "WGSL.h"
#include "WGSLShaderModule.h"
#include <wtf/SetForScope.h>

namespace WGSL {

class PointerRewriter : AST::ScopedVisitor<AST::Expression*> {
    using Base = AST::ScopedVisitor<AST::Expression*>;
    using Base::visit;

public:
    PointerRewriter(ShaderModule& shaderModule)
        : Base()
        , m_shaderModule(shaderModule)
    {
    }

    void run();

    void visit(AST::CompoundStatement&) override;
    void visit(AST::VariableStatement&) override;
    void visit(AST::PhonyAssignmentStatement&) override;
    void visit(AST::LoopStatement&) override;
    void visit(AST::IdentifierExpression&) override;
    void visit(AST::UnaryExpression&) override;

private:
    void rewrite(AST::Statement::List&);

    ShaderModule& m_shaderModule;
    unsigned m_currentStatementIndex { 0 };
    Vector<unsigned> m_indicesToDelete;
};

void PointerRewriter::run()
{
    Base::visit(m_shaderModule);
}

void PointerRewriter::rewrite(AST::Statement::List& statements)
{
    auto indexScope = SetForScope(m_currentStatementIndex, 0);
    auto insertionScope = SetForScope(m_indicesToDelete, Vector<unsigned>());
    ContextScope blockScope(this);

    for (auto& statement : statements) {
        Base::visit(statement);
        ++m_currentStatementIndex;
    }

    for (int i = m_indicesToDelete.size() - 1; i >= 0; --i)
        m_shaderModule.remove(statements, m_indicesToDelete[i]);
}

void PointerRewriter::visit(AST::CompoundStatement& statement)
{
    ContextScope blockScope(this);
    rewrite(statement.statements());
}

void PointerRewriter::visit(AST::VariableStatement& statement)
{
    Base::visit(statement);

    auto& variable = statement.variable();
    auto* initializer = variable.maybeInitializer();
    if (!initializer) {
        introduceVariable(variable.name(), nullptr);
        return;
    }

    auto* pointerType = std::get_if<Types::Pointer>(initializer->inferredType());
    if (!pointerType) {
        introduceVariable(variable.name(), nullptr);
        return;
    }

    introduceVariable(variable.name(), initializer);
    m_indicesToDelete.append(m_currentStatementIndex);
}

void PointerRewriter::visit(AST::PhonyAssignmentStatement& statement)
{
    auto* pointerType = std::get_if<Types::Pointer>(statement.rhs().inferredType());
    if (!pointerType) {
        AST::Visitor::visit(statement);
        return;
    }
    m_indicesToDelete.append(m_currentStatementIndex);
}

void PointerRewriter::visit(AST::LoopStatement& statement)
{
    ContextScope loopScope(this);
    rewrite(statement.body());

    if (auto& continuing = statement.continuing()) {
        ContextScope continuingScope(this);
        rewrite(continuing->body);
    }
}

void PointerRewriter::visit(AST::IdentifierExpression& identifier)
{
    auto* variable = readVariable(identifier.identifier());
    if (!variable || !*variable)
        return;

    auto& identity = m_shaderModule.astBuilder().construct<AST::IdentityExpression>(
        identifier.span(),
        **variable
    );
    m_shaderModule.replace(identifier, identity);
}

void PointerRewriter::visit(AST::UnaryExpression& unary)
{
    Base::visit(unary);

    if (unary.operation() != AST::UnaryOperation::Dereference)
        return;

    AST::Expression* nested = &unary.expression();
    while (is<AST::IdentityExpression>(*nested))
        nested = &downcast<AST::IdentityExpression>(*nested).expression();

    auto* nestedUnary = dynamicDowncast<AST::UnaryExpression>(*nested);
    if (!nestedUnary || nestedUnary->operation() != AST::UnaryOperation::AddressOf)
        return;

    auto& identity = m_shaderModule.astBuilder().construct<AST::IdentityExpression>(
        unary.span(),
        nestedUnary->expression()
    );
    identity.m_inferredType = unary.m_inferredType;
    m_shaderModule.replace(unary, identity);
}

void rewritePointers(ShaderModule& shaderModule)
{
    PointerRewriter(shaderModule).run();
}

} // namespace WGSL
