/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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
#include "antlr/LLkParser.hpp"
#include <stdio.h>

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

ANTLR_USING_NAMESPACE(std)

/**An LL(k) parser.
 *
 * @see antlr.Token
 * @see antlr.TokenBuffer
 * @see antlr.LL1Parser
 */

//	LLkParser(int k_);

LLkParser::LLkParser(const ParserSharedInputState& state, int k_)
: Parser(state), k(k_)
{
}

LLkParser::LLkParser(TokenBuffer& tokenBuf, int k_)
: Parser(tokenBuf), k(k_)
{
}

LLkParser::LLkParser(TokenStream& lexer, int k_)
: Parser(new TokenBuffer(lexer)), k(k_)
{
}

void LLkParser::trace(const char* ee, const char* rname)
{
	traceIndent();

	printf("%s", ((string)ee + rname + ((inputState->guessing>0)?"; [guessing]":"; ")).c_str());

	for (int i = 1; i <= k; i++)
	{
		if (i != 1) {
			printf(", ");
		}
		printf("LA(%d)==", i);

		string temp;

		try {
			temp = LT(i)->getText().c_str();
		}
		catch( ANTLRException& ae )
		{
			temp = "[error: ";
			temp += ae.toString();
			temp += ']';
		}
		printf("%s", temp.c_str());
	}

	printf("\n");
}

void LLkParser::traceIn(const char* rname)
{
	traceDepth++;
	trace("> ",rname);
}

void LLkParser::traceOut(const char* rname)
{
	trace("< ",rname);
	traceDepth--;
}

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif
