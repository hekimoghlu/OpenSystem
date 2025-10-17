/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 28, 2025.
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

#include "ASTVisitor.h"
#include <wtf/StringPrintStream.h>

namespace WGSL {

class ShaderModule;

namespace AST {

class StringDumper final : public Visitor {
    friend struct Indent;
public:
    using Visitor::visit;

    ~StringDumper() = default;

    String toString();

    // Visitor
    void visit(ShaderModule&) override;

    // Directive
    void visit(DiagnosticDirective&) override;

    // Attribute
    void visit(BindingAttribute&) override;
    void visit(BuiltinAttribute&) override;
    void visit(GroupAttribute&) override;
    void visit(LocationAttribute&) override;
    void visit(StageAttribute&) override;
    void visit(WorkgroupSizeAttribute&) override;

    // Declaration
    void visit(Function&) override;
    void visit(Structure&) override;
    void visit(Variable&) override;
    void visit(TypeAlias&) override;

    // Expression
    void visit(AbstractFloatLiteral&) override;
    void visit(AbstractIntegerLiteral&) override;
    void visit(BinaryExpression&) override;
    void visit(BoolLiteral&) override;
    void visit(CallExpression&) override;
    void visit(FieldAccessExpression&) override;
    void visit(Float32Literal&) override;
    void visit(Float16Literal&) override;
    void visit(IdentifierExpression&) override;
    void visit(IndexAccessExpression&) override;
    void visit(PointerDereferenceExpression&) override;
    void visit(Signed32Literal&) override;
    void visit(UnaryExpression&) override;
    void visit(Unsigned32Literal&) override;

    // Statement
    void visit(AssignmentStatement&) override;
    void visit(CallStatement&) override;
    void visit(CompoundAssignmentStatement&) override;
    void visit(CompoundStatement&) override;
    void visit(AST::DecrementIncrementStatement&) override;
    void visit(IfStatement&) override;
    void visit(PhonyAssignmentStatement&) override;
    void visit(ReturnStatement&) override;
    void visit(VariableStatement&) override;
    void visit(ForStatement&) override;

    // Types
    void visit(ArrayTypeExpression&) override;
    void visit(ElaboratedTypeExpression&) override;
    void visit(ReferenceTypeExpression&) override;

    // Values
    void visit(Parameter&) override;

    void visit(StructureMember&) override;

    void visit(VariableQualifier&) override;

private:

    template<typename T, typename J>
    void visitVector(T&, J);

    StringPrintStream m_out;
    String m_indent;
};

template<typename T>
void dumpNode(PrintStream& out, T& node)
{
    StringDumper dumper;
    dumper.visit(node);
    out.print(dumper.toString());
}

MAKE_PRINT_ADAPTOR(ShaderModuleDumper, ShaderModule&, dumpNode);

void dumpAST(ShaderModule&);

} // namespace AST
} // namespace WGSL
