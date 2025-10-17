/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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
#include "ASTStringDumper.h"

#include "AST.h"
#include "WGSLShaderModule.h"
#include <wtf/DataLog.h>
#include <wtf/SetForScope.h>
#include <wtf/text/MakeString.h>

namespace WGSL::AST {

struct Indent {
    Indent(StringDumper& dumper)
        : scope(dumper.m_indent, makeString(dumper.m_indent, "    "_s))
    { }
    SetForScope<String> scope;
};

static Indent bumpIndent(StringDumper& dumper)
{
    return Indent(dumper);
}

template<typename T, typename J>
void StringDumper::visitVector(T& nodes, J joiner)
{
    if (nodes.isEmpty())
        return;

    visit(nodes[0]);
    for (size_t n = 1; n < nodes.size(); ++n) {
        m_out.print(joiner);
        visit(nodes[n]);
    }
}

String StringDumper::toString()
{
    return m_out.toString();
}

// Shader Module
void StringDumper::visit(ShaderModule& shaderModule)
{
    for (auto& directive : shaderModule.directives())
        visit(directive);
    if (!shaderModule.directives().isEmpty())
        m_out.print("\n\n");

    for (auto& declaration : shaderModule.declarations()) {
        AST::Visitor::visit(declaration);
        m_out.print("\n");
    }
}

void StringDumper::visit(DiagnosticDirective&)
{
    // FIXME: we still don't do anything with diagnostics
}

// Attribute
void StringDumper::visit(BindingAttribute& binding)
{
    m_out.print("@binding(");
    visit(binding.binding());
    m_out.print(")");
}

void StringDumper::visit(BuiltinAttribute& builtin)
{
    m_out.print("@builtin(", builtin.builtin(), ")");
}

void StringDumper::visit(GroupAttribute& group)
{
    m_out.print("@group(");
    visit(group.group());
    m_out.print(")");
}

void StringDumper::visit(LocationAttribute& location)
{
    m_out.print("@location(");
    visit(location.location());
    m_out.print(")");
}

void StringDumper::visit(StageAttribute& stage)
{
    m_out.print("@", stage.stage());
}

void StringDumper::visit(WorkgroupSizeAttribute& workgroupSize)
{
    m_out.print("@workgroup_size(");
    visit(workgroupSize.x());
    if (auto* y = workgroupSize.maybeY()) {
        m_out.print(", ");
        visit(*y);
        if (auto* z = workgroupSize.maybeZ()) {
            m_out.print(", ");
            visit(*z);
        }
    }
    m_out.print(")");
}

// Declaration
void StringDumper::visit(Function& function)
{
    m_out.print(m_indent);
    if (!function.attributes().isEmpty()) {
        visitVector(function.attributes(), " ");
        m_out.print("\n", m_indent);
    }
    m_out.print("fn ", function.name(), "(");
    if (!function.parameters().isEmpty()) {
        m_out.print("\n");
        {
            auto indent = bumpIndent(*this);
            visitVector(function.parameters(), "\n");
        }
        m_out.print("\n", m_indent);
    }
    m_out.print(")");
    if (function.maybeReturnType()) {
        m_out.print(" -> ");
        visitVector(function.returnAttributes(), " ");
        m_out.print(" ");
        visit(*function.maybeReturnType());
    }
    m_out.print("\n", m_indent);
    visit(function.body());
}

void StringDumper::visit(Structure& structure)
{
    m_out.print(m_indent);
    if (!structure.attributes().isEmpty()) {
        visitVector(structure.attributes(), " ");
        m_out.print("\n", m_indent);
    }
    m_out.print("struct ", structure.name(), " {");
    if (!structure.members().isEmpty()) {
        m_out.print("\n");
        {
            auto indent = bumpIndent(*this);
            visitVector(structure.members(), ",\n");
        }
        m_out.print("\n", m_indent);
    }
    m_out.print("}\n");
}

void StringDumper::visit(Variable& variable)
{
    if (!variable.attributes().isEmpty()) {
        visitVector(variable.attributes(), " ");
        m_out.print(" ");
    }
    m_out.print("var");
    if (variable.maybeQualifier())
        visit(*variable.maybeQualifier());
    m_out.print(" ", variable.name());
    if (variable.maybeTypeName()) {
        m_out.print(": ");
        visit(*variable.maybeTypeName());
    }
    if (variable.maybeInitializer()) {
        m_out.print(" = ");
        visit(*variable.maybeInitializer());
    }
    m_out.print(";");
}

void StringDumper::visit(TypeAlias& alias)
{
    m_out.print(m_indent);
    m_out.print("alias ", alias.name(), " = ");
    visit(alias.type());
    m_out.print(";");
}

// Expression
void StringDumper::visit(AbstractFloatLiteral& literal)
{
    m_out.print(literal.value());
}

void StringDumper::visit(AbstractIntegerLiteral& literal)
{
    m_out.print(literal.value());
}

void StringDumper::visit(IndexAccessExpression& arrayAccess)
{
    visit(arrayAccess.base());
    m_out.print("[");
    visit(arrayAccess.index());
    m_out.print("]");
}

void StringDumper::visit(BoolLiteral& literal)
{
    m_out.print(literal.value() ? "true": "false");
}

void StringDumper::visit(CallExpression& expression)
{
    visit(expression.target());
    m_out.print("(");
    if (!expression.arguments().isEmpty()) {
        auto indent = bumpIndent(*this);
        visitVector(expression.arguments(), ", ");
    }
    m_out.print(")");
}

void StringDumper::visit(Float32Literal& literal)
{
    m_out.print(literal.value(), "f");
}

void StringDumper::visit(Float16Literal& literal)
{
    m_out.print(String::number(literal.value()), "h");
}

void StringDumper::visit(IdentifierExpression& identifier)
{
    m_out.print(identifier.identifier());
}

void StringDumper::visit(Signed32Literal& literal)
{
    m_out.print(literal.value(), "i");
}

void StringDumper::visit(FieldAccessExpression& fieldAccess)
{
    visit(fieldAccess.base());
    m_out.print(".", fieldAccess.fieldName());
}

void StringDumper::visit(Unsigned32Literal& literal)
{
    m_out.print(literal.value(), "u");
}

void StringDumper::visit(UnaryExpression& expression)
{
    m_out.print(expression.operation());
    visit(expression.expression());
}

void StringDumper::visit(BinaryExpression& expression)
{
    visit(expression.leftExpression());
    m_out.print(" ", expression.operation(), " ");
    visit(expression.rightExpression());
}

void StringDumper::visit(PointerDereferenceExpression& pointerDereference)
{
    m_out.print("(*");
    visit(pointerDereference.target());
    m_out.print(")");
}

// Statement
void StringDumper::visit(AssignmentStatement& statement)
{
    m_out.print(m_indent);
    visit(statement.lhs());
    m_out.print(" = ");
    visit(statement.rhs());
    m_out.print(";");
}

void StringDumper::visit(CallStatement& statement)
{
    visit(statement.call());
    m_out.print(";");
}

void StringDumper::visit(CompoundAssignmentStatement& statement)
{
    m_out.print(m_indent);
    visit(statement.leftExpression());
    m_out.print(" ", statement.operation(), "= ");
    visit(statement.rightExpression());
    m_out.print(";");
}

void StringDumper::visit(CompoundStatement& block)
{
    m_out.print(m_indent, "{");
    if (!block.statements().isEmpty()) {
        {
            auto indent = bumpIndent(*this);
            m_out.print("\n");
            visitVector(block.statements(), "\n");
        }
        m_out.print("\n", m_indent);
    }
    m_out.print("}\n");
}

void StringDumper::visit(DecrementIncrementStatement& statement)
{
    m_out.print(m_indent);
    visit(statement.expression());
    m_out.print(statement.operation(), ";");
}


void StringDumper::visit(IfStatement& statement)
{
    m_out.print(m_indent, "if ");
    visit(statement.test());
    m_out.print("\n");
    visit(statement.trueBody());
    if (statement.maybeFalseBody()) {
        m_out.print(m_indent, "else");
        if (is<IfStatement>(*statement.maybeFalseBody()))
            m_out.print(" ");
        else
            m_out.print("\n");
        visit(*statement.maybeFalseBody());
    }
}

void StringDumper::visit(PhonyAssignmentStatement& statement)
{
    m_out.print(m_indent);
    m_out.print("_ = ");
    visit(statement.rhs());
    m_out.print(";");
}

void StringDumper::visit(ReturnStatement& statement)
{
    m_out.print(m_indent, "return");
    if (statement.maybeExpression()) {
        m_out.print(" ");
        visit(*statement.maybeExpression());
    }
    m_out.print(";");
}

void StringDumper::visit(VariableStatement& statement)
{
    m_out.print(m_indent);
    visit(statement.variable());
}

void StringDumper::visit(ForStatement& statement)
{
    m_out.print("for (");
    if (auto* initializer = statement.maybeInitializer())
        visit(*initializer);
    m_out.print(";");
    if (auto* test = statement.maybeTest())
        visit(*test);
    m_out.print(";");
    if (auto* update = statement.maybeUpdate())
        visit(*update);
    m_out.print(")");
    visit(statement.body());
}

// Types
void StringDumper::visit(ArrayTypeExpression& type)
{
    m_out.print("array");
    if (type.maybeElementType()) {
        m_out.print("<");
        visit(*type.maybeElementType());
        if (type.maybeElementCount()) {
            m_out.print(", ");
            visit(*type.maybeElementCount());
        }
        m_out.print(">");
    }
}

void StringDumper::visit(ElaboratedTypeExpression& type)
{
    m_out.print(type.base(), "<");
    bool first = true;
    for (auto& argument : type.arguments()) {
        if (!first)
            m_out.print(", ");
        first = false;
        visit(argument);
    }
    m_out.print(">");
}

void StringDumper::visit(ReferenceTypeExpression& type)
{
    visit(type.type());
    m_out.print("&");
}

void StringDumper::visit(Parameter& parameter)
{
    m_out.print(m_indent);
    if (!parameter.attributes().isEmpty()) {
        visitVector(parameter.attributes(), " ");
        m_out.print(" ");
    }
    m_out.print(parameter.name(), ": ");
    visit(parameter.typeName());
}

void StringDumper::visit(StructureMember& member)
{
    m_out.print(m_indent);
    if (!member.attributes().isEmpty()) {
        visitVector(member.attributes(), " ");
        m_out.print(" ");
    }
    m_out.print(member.name(), ": ");
    visit(member.type());
}

void StringDumper::visit(VariableQualifier& qualifier)
{
    m_out.print("<", qualifier.addressSpace(), ",", qualifier.accessMode(), ">");
}

void dumpAST(ShaderModule& shaderModule)
{
    dataLogLn(ShaderModuleDumper(shaderModule));
}

} // namespace WGSL::AST
