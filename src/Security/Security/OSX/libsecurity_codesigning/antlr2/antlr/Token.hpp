/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 26, 2025.
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

#ifndef INC_Token_hpp__
#define INC_Token_hpp__

/* ANTLR Translator Generator
 * Project led by Terence Parr at http://www.jGuru.com
 * Software rights: http://www.antlr.org/license.html
 *
 * $Id: //depot/code/org.antlr/release/antlr-2.7.7/lib/cpp/antlr/Token.hpp#2 $
 */

#include <antlr/config.hpp>
#include <antlr/TokenRefCount.hpp>
#include <string>

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

struct TokenRef;

/** A token is minimally a token type.  Subclasses can add the text matched
 *  for the token and line info.
 */
class ANTLR_API Token
{
public:
	// constants
#ifndef NO_STATIC_CONSTS
	static const int MIN_USER_TYPE = 4;
	static const int NULL_TREE_LOOKAHEAD = 3;
	static const int INVALID_TYPE = 0;
	static const int EOF_TYPE = 1;
	static const int SKIP = -1;
#else
	enum {
		MIN_USER_TYPE = 4,
		NULL_TREE_LOOKAHEAD = 3,
		INVALID_TYPE = 0,
		EOF_TYPE = 1,
		SKIP = -1
	};
#endif

	Token()
	: ref(0)
	, type(INVALID_TYPE)
	{
	}
	Token(int t)
	: ref(0)
	, type(t)
	{
	}
	Token(int t, const ANTLR_USE_NAMESPACE(std)string& txt)
	: ref(0)
	, type(t)
	{
		setText(txt);
	}
	virtual ~Token()
	{
	}

	virtual int getColumn() const;
	virtual int getLine() const;
	virtual ANTLR_USE_NAMESPACE(std)string getText() const;
	virtual const ANTLR_USE_NAMESPACE(std)string& getFilename() const;
	virtual int getType() const;

	virtual void setColumn(int c);

	virtual void setLine(int l);
	virtual void setText(const ANTLR_USE_NAMESPACE(std)string& t);
	virtual void setType(int t);

	virtual void setFilename( const std::string& file );

	virtual ANTLR_USE_NAMESPACE(std)string toString() const;

private:
	friend struct TokenRef;
	TokenRef* ref;

	int type; 							///< the type of the token

	Token(RefToken other);
	Token& operator=(const Token& other);
	Token& operator=(RefToken other);

	Token(const Token&);
};

extern ANTLR_API RefToken nullToken;

#ifdef NEEDS_OPERATOR_LESS_THAN
// RK: Added after 2.7.2 previously it was undefined.
// AL: what to return if l and/or r point to nullToken???
inline bool operator<( RefToken l, RefToken r )
{
	return nullToken == l ? ( nullToken == r ? false : true ) : l->getType() < r->getType();
}
#endif

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif

#endif //INC_Token_hpp__
