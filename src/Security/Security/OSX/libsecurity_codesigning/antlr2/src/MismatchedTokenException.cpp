/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 21, 2024.
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
#include "antlr/MismatchedTokenException.hpp"
#include "antlr/String.hpp"

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

MismatchedTokenException::MismatchedTokenException()
  : RecognitionException("Mismatched Token: expecting any AST node","<AST>",-1,-1)
  , token(0)
  , node(nullASTptr)
  , tokenNames(0)
  , numTokens(0)
{
}

// Expected range / not range
MismatchedTokenException::MismatchedTokenException(
	const char* const* tokenNames_,
	const int numTokens_,
	RefAST node_,
	int lower,
	int upper_,
	bool matchNot
) : RecognitionException("Mismatched Token","<AST>",-1,-1)
  , token(0)
  , node(node_)
  , tokenText( (node_ ? node_->toString(): ANTLR_USE_NAMESPACE(std)string("<empty tree>")) )
  , mismatchType(matchNot ? NOT_RANGE : RANGE)
  , expecting(lower)
  , upper(upper_)
  , tokenNames(tokenNames_)
  , numTokens(numTokens_)
{
}

// Expected token / not token
MismatchedTokenException::MismatchedTokenException(
	const char* const* tokenNames_,
	const int numTokens_,
	RefAST node_,
	int expecting_,
	bool matchNot
) : RecognitionException("Mismatched Token","<AST>",-1,-1)
  , token(0)
  , node(node_)
  , tokenText( (node_ ? node_->toString(): ANTLR_USE_NAMESPACE(std)string("<empty tree>")) )
  , mismatchType(matchNot ? NOT_TOKEN : TOKEN)
  , expecting(expecting_)
  , tokenNames(tokenNames_)
  , numTokens(numTokens_)
{
}

// Expected BitSet / not BitSet
MismatchedTokenException::MismatchedTokenException(
	const char* const* tokenNames_,
	const int numTokens_,
	RefAST node_,
	BitSet set_,
	bool matchNot
) : RecognitionException("Mismatched Token","<AST>",-1,-1)
  , token(0)
  , node(node_)
  , tokenText( (node_ ? node_->toString(): ANTLR_USE_NAMESPACE(std)string("<empty tree>")) )
  , mismatchType(matchNot ? NOT_SET : SET)
  , set(set_)
  , tokenNames(tokenNames_)
  , numTokens(numTokens_)
{
}

// Expected range / not range
MismatchedTokenException::MismatchedTokenException(
	const char* const* tokenNames_,
	const int numTokens_,
	RefToken token_,
	int lower,
	int upper_,
	bool matchNot,
	const ANTLR_USE_NAMESPACE(std)string& fileName_
) : RecognitionException("Mismatched Token",fileName_,token_->getLine(),token_->getColumn())
  , token(token_)
  , node(nullASTptr)
  , tokenText(token_->getText())
  , mismatchType(matchNot ? NOT_RANGE : RANGE)
  , expecting(lower)
  , upper(upper_)
  , tokenNames(tokenNames_)
  , numTokens(numTokens_)
{
}

// Expected token / not token
MismatchedTokenException::MismatchedTokenException(
	const char* const* tokenNames_,
	const int numTokens_,
	RefToken token_,
	int expecting_,
	bool matchNot,
	const ANTLR_USE_NAMESPACE(std)string& fileName_
) : RecognitionException("Mismatched Token",fileName_,token_->getLine(),token_->getColumn())
  , token(token_)
  , node(nullASTptr)
  , tokenText(token_->getText())
  , mismatchType(matchNot ? NOT_TOKEN : TOKEN)
  , expecting(expecting_)
  , tokenNames(tokenNames_)
  , numTokens(numTokens_)
{
}

// Expected BitSet / not BitSet
MismatchedTokenException::MismatchedTokenException(
	const char* const* tokenNames_,
	const int numTokens_,
	RefToken token_,
	BitSet set_,
	bool matchNot,
	const ANTLR_USE_NAMESPACE(std)string& fileName_
) : RecognitionException("Mismatched Token",fileName_,token_->getLine(),token_->getColumn())
  , token(token_)
  , node(nullASTptr)
  , tokenText(token_->getText())
  , mismatchType(matchNot ? NOT_SET : SET)
  , set(set_)
  , tokenNames(tokenNames_)
  , numTokens(numTokens_)
{
}

ANTLR_USE_NAMESPACE(std)string MismatchedTokenException::getMessage() const
{
	ANTLR_USE_NAMESPACE(std)string s;
	switch (mismatchType) {
	case TOKEN:
		s += "expecting " + tokenName(expecting) + ", found '" + tokenText + "'";
		break;
	case NOT_TOKEN:
		s += "expecting anything but " + tokenName(expecting) + "; got it anyway";
		break;
	case RANGE:
		s += "expecting token in range: " + tokenName(expecting) + ".." + tokenName(upper) + ", found '" + tokenText + "'";
		break;
	case NOT_RANGE:
		s += "expecting token NOT in range: " + tokenName(expecting) + ".." + tokenName(upper) + ", found '" + tokenText + "'";
		break;
	case SET:
	case NOT_SET:
		{
			s += ANTLR_USE_NAMESPACE(std)string("expecting ") + (mismatchType == NOT_SET ? "NOT " : "") + "one of (";
			ANTLR_USE_NAMESPACE(std)vector<unsigned int> elems = set.toArray();
			for ( unsigned int i = 0; i < elems.size(); i++ )
			{
				s += " ";
				s += tokenName(elems[i]);
			}
			s += "), found '" + tokenText + "'";
		}
		break;
	default:
		s = RecognitionException::getMessage();
		break;
	}
	return s;
}

ANTLR_USE_NAMESPACE(std)string MismatchedTokenException::tokenName(int tokenType) const
{
	if (tokenType == Token::INVALID_TYPE)
		return "<Set of tokens>";
	else if (tokenType < 0 || tokenType >= numTokens)
		return ANTLR_USE_NAMESPACE(std)string("<") + tokenType + ">";
	else
		return tokenNames[tokenType];
}

#ifndef NO_STATIC_CONSTS
const int MismatchedTokenException::TOKEN;
const int MismatchedTokenException::NOT_TOKEN;
const int MismatchedTokenException::RANGE;
const int MismatchedTokenException::NOT_RANGE;
const int MismatchedTokenException::SET;
const int MismatchedTokenException::NOT_SET;
#endif

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif
