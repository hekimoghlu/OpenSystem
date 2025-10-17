/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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

#include "ExceptionOr.h"
#include "XPathPredicate.h"

union YYSTYPE;

namespace WebCore {

class XPathNSResolver;

namespace XPath {

class Parser {
    WTF_MAKE_NONCOPYABLE(Parser);
public:
    static ExceptionOr<std::unique_ptr<Expression>> parseStatement(const String& statement, RefPtr<XPathNSResolver>&&);

    int lex(YYSTYPE&);
    bool expandQualifiedName(const String& qualifiedName, AtomString& localName, AtomString& namespaceURI);
    void setParseResult(std::unique_ptr<Expression>&& expression) { m_result = WTFMove(expression); }

private:
    Parser(const String&, RefPtr<XPathNSResolver>&&);

    struct Token;

    bool isBinaryOperatorContext() const;

    void skipWS();
    Token makeTokenAndAdvance(int type, int advance = 1);
    Token makeTokenAndAdvance(int type, NumericOp::Opcode, int advance = 1);
    Token makeTokenAndAdvance(int type, EqTestOp::Opcode, int advance = 1);
    char peekAheadHelper();
    char peekCurHelper();

    Token lexString();
    Token lexNumber();
    bool lexNCName(String&);
    bool lexQName(String&);

    Token nextToken();
    Token nextTokenInternal();

    const String& m_data;
    RefPtr<XPathNSResolver> m_resolver;

    unsigned m_nextPos { 0 };
    int m_lastTokenType { 0 };

    std::unique_ptr<Expression> m_result;
    bool m_sawNamespaceError { false };
};

} }
