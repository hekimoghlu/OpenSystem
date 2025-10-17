/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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

#ifndef INC_TokenWithIndex_hpp__
#define INC_TokenWithIndex_hpp__

/* ANTLR Translator Generator
 * Project led by Terence Parr at http://www.jGuru.com
 * Software rights: http://www.antlr.org/license.html
 *
 * $Id:$
 */

#include <antlr/config.hpp>
#include <antlr/CommonToken.hpp>
#include <antlr/String.hpp>

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

class ANTLR_API TokenWithIndex : public ANTLR_USE_NAMESPACE(antlr)CommonToken {
public:
	// static size_t count;
	TokenWithIndex() : CommonToken(), index(0)
	{
		// std::cout << __PRETTY_FUNCTION__ << std::endl;
		// count++;
	}
	TokenWithIndex(int t, const ANTLR_USE_NAMESPACE(std)string& txt)
	: CommonToken(t,txt)
	, index(0)
	{
		// std::cout << __PRETTY_FUNCTION__ << std::endl;
		// count++;
	}
	TokenWithIndex(const ANTLR_USE_NAMESPACE(std)string& s)
	: CommonToken(s)
	, index(0)
	{
		// std::cout << __PRETTY_FUNCTION__ << std::endl;
		// count++;
	}
	~TokenWithIndex()
	{
		// count--;
	}
	void setIndex( size_t idx )
	{
		index = idx;
	}
	size_t getIndex( void ) const
	{
		return index;
	}

	ANTLR_USE_NAMESPACE(std)string toString() const
	{
		return ANTLR_USE_NAMESPACE(std)string("[")+
			index+
			":\""+
			getText()+"\",<"+
			getType()+">,line="+
			getLine()+",column="+
			getColumn()+"]";
	}

	static RefToken factory()
	{
		return RefToken(new TokenWithIndex());
	}

protected:
	size_t index;

private:
	TokenWithIndex(const TokenWithIndex&);
	const TokenWithIndex& operator=(const TokenWithIndex&);
};

typedef TokenRefCount<TokenWithIndex> RefTokenWithIndex;

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif

#endif //INC_CommonToken_hpp__
