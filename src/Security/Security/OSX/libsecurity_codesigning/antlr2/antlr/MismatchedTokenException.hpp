/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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

#ifndef INC_MismatchedTokenException_hpp__
#define INC_MismatchedTokenException_hpp__

/* ANTLR Translator Generator
 * Project led by Terence Parr at http://www.jGuru.com
 * Software rights: http://www.antlr.org/license.html
 *
 * $Id: //depot/code/org.antlr/release/antlr-2.7.7/lib/cpp/antlr/MismatchedTokenException.hpp#2 $
 */

#include <antlr/config.hpp>
#include <antlr/RecognitionException.hpp>
#include <antlr/BitSet.hpp>
#include <antlr/Token.hpp>
#include <antlr/AST.hpp>
#include <vector>

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

class ANTLR_API MismatchedTokenException : public RecognitionException {
public:
	MismatchedTokenException();

	/// Expected range / not range
	MismatchedTokenException(
		const char* const* tokenNames_,
		const int numTokens_,
		RefAST node_,
		int lower,
		int upper_,
		bool matchNot
	);

	// Expected token / not token
	MismatchedTokenException(
		const char* const* tokenNames_,
		const int numTokens_,
		RefAST node_,
		int expecting_,
		bool matchNot
	);

	// Expected BitSet / not BitSet
	MismatchedTokenException(
		const char* const* tokenNames_,
		const int numTokens_,
		RefAST node_,
		BitSet set_,
		bool matchNot
	);

	// Expected range / not range
	MismatchedTokenException(
		const char* const* tokenNames_,
		const int numTokens_,
		RefToken token_,
		int lower,
		int upper_,
		bool matchNot,
		const ANTLR_USE_NAMESPACE(std)string& fileName_
	);

	// Expected token / not token
	MismatchedTokenException(
		const char* const* tokenNames_,
		const int numTokens_,
		RefToken token_,
		int expecting_,
		bool matchNot,
		const ANTLR_USE_NAMESPACE(std)string& fileName_
	);

	// Expected BitSet / not BitSet
	MismatchedTokenException(
		const char* const* tokenNames_,
		const int numTokens_,
		RefToken token_,
		BitSet set_,
		bool matchNot,
		const ANTLR_USE_NAMESPACE(std)string& fileName_
	);
	~MismatchedTokenException() _NOEXCEPT {}

	/**
	 * Returns a clean error message (no line number/column information)
	 */
	ANTLR_USE_NAMESPACE(std)string getMessage() const;

public:
	/// The token that was encountered
	const RefToken token;
	/// The offending AST node if tree walking
	const RefAST node;
	/// taken from node or token object
	ANTLR_USE_NAMESPACE(std)string tokenText;

	/// Types of tokens
#ifndef NO_STATIC_CONSTS
	static const int TOKEN = 1;
	static const int NOT_TOKEN = 2;
	static const int RANGE = 3;
	static const int NOT_RANGE = 4;
	static const int SET = 5;
	static const int NOT_SET = 6;
#else
	enum {
		TOKEN = 1,
		NOT_TOKEN = 2,
		RANGE = 3,
		NOT_RANGE = 4,
		SET = 5,
		NOT_SET = 6
	};
#endif

public:
	/// One of the above
	int mismatchType;

	/// For TOKEN/NOT_TOKEN and RANGE/NOT_RANGE
	int expecting;

	/// For RANGE/NOT_RANGE (expecting is lower bound of range)
	int upper;

	/// For SET/NOT_SET
	BitSet set;

private:
	/// Token names array for formatting
	const char* const* tokenNames;
	/// Max number of tokens in tokenNames
	const int numTokens;
	/// Return token name for tokenType
	ANTLR_USE_NAMESPACE(std)string tokenName(int tokenType) const;
};

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif

#endif //INC_MismatchedTokenException_hpp__
