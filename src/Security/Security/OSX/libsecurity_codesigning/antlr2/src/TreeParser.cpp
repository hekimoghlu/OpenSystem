/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 18, 2023.
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
#include "antlr/TreeParser.hpp"
#include "antlr/ASTNULLType.hpp"
#include <stdio.h>

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

/** The AST Null object; the parsing cursor is set to this when
 *  it is found to be null.  This way, we can test the
 *  token type of a node without having to have tests for null
 *  everywhere.
 */
[[clang::no_destroy]] RefAST TreeParser::ASTNULL(new ASTNULLType);

/** Parser error-reporting function can be overridden in subclass */
void TreeParser::reportError(const RecognitionException& ex)
{
	fprintf(stderr, "%s", (ex.toString() + "\n").c_str());
}

/** Parser error-reporting function can be overridden in subclass */
void TreeParser::reportError(const ANTLR_USE_NAMESPACE(std)string& s)
{
	fprintf(stderr, "%s", ("error: " + s + "\n").c_str());
}

/** Parser warning-reporting function can be overridden in subclass */
void TreeParser::reportWarning(const ANTLR_USE_NAMESPACE(std)string& s)
{
	fprintf(stderr, "%s", ("warning: " + s + "\n").c_str());
}

/** Procedure to write out an indent for traceIn and traceOut */
void TreeParser::traceIndent()
{
	for( int i = 0; i < traceDepth; i++ )
		printf(" ");
}

void TreeParser::traceIn(const char* rname, RefAST t)
{
	traceDepth++;
	traceIndent();

	printf("%s",((ANTLR_USE_NAMESPACE(std)string)"> " + rname
			+ "(" + (t ? t->toString().c_str() : "null") + ")"
			+ ((inputState->guessing>0)?" [guessing]":"")
			+ "\n").c_str());
}

void TreeParser::traceOut(const char* rname, RefAST t)
{
	traceIndent();

	printf("%s",((ANTLR_USE_NAMESPACE(std)string)"< " + rname
			+ "(" + (t ? t->toString().c_str() : "null") + ")"
			+ ((inputState->guessing>0)?" [guessing]":"")
			+ "\n").c_str());

	traceDepth--;
}

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif
