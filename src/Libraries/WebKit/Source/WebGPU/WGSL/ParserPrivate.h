/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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

#include "ASTAttribute.h"
#include "ASTBuilder.h"
#include "ASTConstAssert.h"
#include "ASTExpression.h"
#include "ASTForward.h"
#include "ASTFunction.h"
#include "ASTStatement.h"
#include "ASTStructure.h"
#include "ASTTypeAlias.h"
#include "ASTVariable.h"
#include "CompilationMessage.h"
#include "Lexer.h"
#include "WGSLShaderModule.h"
#include <wtf/Ref.h>

namespace WGSL {

class ShaderModule;

template<typename Lexer>
class Parser {
public:
    Parser(ShaderModule& shaderModule, Lexer& lexer)
        : m_shaderModule(shaderModule)
        , m_builder(shaderModule.astBuilder())
        , m_lexer(lexer)
        , m_tokens(m_lexer.lex())
        , m_current(m_tokens[0])
        , m_currentPosition({ m_current.span.line, m_current.span.lineOffset, m_current.span.offset })
    {
    }

    Result<void> parseShader();

    void maybeSplitToken(unsigned index);
    void disambiguateTemplates();

    // AST::<type>::Ref whenever it can return multiple types.
    Result<AST::Identifier> parseIdentifier();
    Result<void> parseEnableDirective();
    Result<void> parseRequireDirective();
    Result<AST::Declaration::Ref> parseDeclaration();
    Result<AST::ConstAssert::Ref> parseConstAssert();
    Result<AST::Attribute::List> parseAttributes();
    Result<AST::Attribute::Ref> parseAttribute();
    Result<AST::Structure::Ref> parseStructure(AST::Attribute::List&&);
    Result<std::reference_wrapper<AST::StructureMember>> parseStructureMember();
    Result<AST::Expression::Ref> parseTypeName();
    Result<AST::Expression::Ref> parseTypeNameAfterIdentifier(AST::Identifier&&, SourcePosition start);
    Result<AST::Expression::Ref> parseArrayType();
    Result<AST::Variable::Ref> parseVariable();
    Result<AST::Variable::Ref> parseVariableWithAttributes(AST::Attribute::List&&);
    Result<AST::VariableQualifier::Ref> parseVariableQualifier();
    Result<AddressSpace> parseAddressSpace();
    Result<AccessMode> parseAccessMode();
    Result<AST::TypeAlias::Ref> parseTypeAlias();
    Result<AST::Function::Ref> parseFunction(AST::Attribute::List&&);
    Result<std::reference_wrapper<AST::Parameter>> parseParameter();
    Result<AST::Statement::Ref> parseStatement();
    Result<AST::CompoundStatement::Ref> parseCompoundStatement();
    Result<AST::Statement::Ref> parseIfStatement();
    Result<AST::Statement::Ref> parseIfStatementWithAttributes(AST::Attribute::List&&, SourcePosition _startOfElementPosition);
    Result<AST::Statement::Ref> parseForStatement();
    Result<AST::Statement::Ref> parseLoopStatement();
    Result<AST::Statement::Ref> parseSwitchStatement();
    Result<AST::Statement::Ref> parseWhileStatement();
    Result<AST::Statement::Ref> parseReturnStatement();
    Result<AST::Statement::Ref> parseVariableUpdatingStatement();
    Result<AST::Statement::Ref> parseVariableUpdatingStatement(AST::Expression::Ref&&);
    Result<AST::Expression::Ref> parseShortCircuitExpression(AST::Expression::Ref&&, TokenType, AST::BinaryOperation);
    Result<AST::Expression::Ref> parseRelationalExpression();
    Result<AST::Expression::Ref> parseRelationalExpressionPostUnary(AST::Expression::Ref&& lhs);
    Result<AST::Expression::Ref> parseShiftExpression();
    Result<AST::Expression::Ref> parseShiftExpressionPostUnary(AST::Expression::Ref&& lhs);
    Result<AST::Expression::Ref> parseAdditiveExpressionPostUnary(AST::Expression::Ref&& lhs);
    Result<AST::Expression::Ref> parseBitwiseExpressionPostUnary(AST::Expression::Ref&& lhs);
    Result<AST::Expression::Ref> parseMultiplicativeExpressionPostUnary(AST::Expression::Ref&& lhs);
    Result<AST::Expression::Ref> parseUnaryExpression();
    Result<AST::Expression::Ref> parseSingularExpression();
    Result<AST::Expression::Ref> parsePostfixExpression(AST::Expression::Ref&& base, SourcePosition startPosition);
    Result<AST::Expression::Ref> parsePrimaryExpression();
    Result<AST::Expression::Ref> parseExpression();
    Result<AST::Expression::Ref> parseLHSExpression();
    Result<AST::Expression::Ref> parseCoreLHSExpression();
    Result<AST::Expression::List> parseArgumentExpressionList();
    Result<AST::Diagnostic> parseDiagnostic();

private:
    Expected<Token, TokenType> consumeType(TokenType);
    template<TokenType... TTs> Expected<Token, TokenType> consumeTypes();

    void consume();

    Token& current() { return m_current; }

    ShaderModule& m_shaderModule;
    AST::Builder& m_builder;
    Lexer& m_lexer;
    Vector<Token> m_tokens;
    unsigned m_currentTokenIndex { 0 };
    unsigned m_parseDepth { 0 };
    unsigned m_compositeTypeDepth { 0 };
    Token m_current;
    SourcePosition m_currentPosition;
};

} // namespace WGSL
