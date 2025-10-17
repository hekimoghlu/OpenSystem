/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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
#include "antlr/Parser.hpp"

#include <stdio.h>

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

/** A generic ANTLR parser (LL(k) for k>=1) containing a bunch of
 * utility routines useful at any lookahead depth.  We distinguish between
 * the LL(1) and LL(k) parsers because of efficiency.  This may not be
 * necessary in the near future.
 *
 * Each parser object contains the state of the parse including a lookahead
 * cache (the form of which is determined by the subclass), whether or
 * not the parser is in guess mode, where tokens come from, etc...
 *
 * <p>
 * During <b>guess</b> mode, the current lookahead token(s) and token type(s)
 * cache must be saved because the token stream may not have been informed
 * to save the token (via <tt>mark</tt>) before the <tt>try</tt> block.
 * Guessing is started by:
 * <ol>
 * <li>saving the lookahead cache.
 * <li>marking the current position in the TokenBuffer.
 * <li>increasing the guessing level.
 * </ol>
 *
 * After guessing, the parser state is restored by:
 * <ol>
 * <li>restoring the lookahead cache.
 * <li>rewinding the TokenBuffer.
 * <li>decreasing the guessing level.
 * </ol>
 *
 * @see antlr.Token
 * @see antlr.TokenBuffer
 * @see antlr.TokenStream
 * @see antlr.LL1Parser
 * @see antlr.LLkParser
 */

bool DEBUG_PARSER = false;

/** Parser error-reporting function can be overridden in subclass */
void Parser::reportError(const RecognitionException& ex)
{
	fprintf(stderr, "%s", (ex.toString() + "\n").c_str());
}

/** Parser error-reporting function can be overridden in subclass */
void Parser::reportError(const ANTLR_USE_NAMESPACE(std)string& s)
{
	if ( getFilename()=="" )
		fprintf(stderr, "%s", ("error: " + s + "\n").c_str());
	else
		fprintf(stderr, "%s", (getFilename() + ": error: " + s + "\n").c_str());
}

/** Parser warning-reporting function can be overridden in subclass */
void Parser::reportWarning(const ANTLR_USE_NAMESPACE(std)string& s)
{
    if ( getFilename()=="" )
        fprintf(stderr, "%s", ("warning: " + s + "\n").c_str());
    else
        fprintf(stderr, "%s", (getFilename() + ": warning: " + s + "\n").c_str());
}

/** Set or change the input token buffer */
//	void setTokenBuffer(TokenBuffer<Token>* t);

void Parser::traceIndent()
{
	for( int i = 0; i < traceDepth; i++ )
		printf(" ");
}

void Parser::traceIn(const char* rname)
{
	traceDepth++;

	for( int i = 0; i < traceDepth; i++ )
		printf(" ");

	printf("%s",((ANTLR_USE_NAMESPACE(std)string)"> " + rname
		+ "; LA(1)==" + LT(1)->getText()
		+	((inputState->guessing>0)?" [guessing]":"")
		+ "\n").c_str());
}

void Parser::traceOut(const char* rname)
{
	for( int i = 0; i < traceDepth; i++ )
        printf(" ");

	printf("%s",((ANTLR_USE_NAMESPACE(std)string)"< " + rname
		+ "; LA(1)==" + LT(1)->getText()
		+	((inputState->guessing>0)?" [guessing]":"")
		+ "\n").c_str());

	traceDepth--;
}

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif
