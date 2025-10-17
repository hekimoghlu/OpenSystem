/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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

#ifndef INC_CommonASTWithHiddenTokens_hpp__
#define INC_CommonASTWithHiddenTokens_hpp__

/* ANTLR Translator Generator
 * Project led by Terence Parr at http://www.jGuru.com
 * Software rights: http://www.antlr.org/license.html
 *
 * $Id: //depot/code/org.antlr/release/antlr-2.7.7/lib/cpp/antlr/CommonASTWithHiddenTokens.hpp#2 $
 */

#include <antlr/config.hpp>
#include <antlr/CommonAST.hpp>

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

/** A CommonAST whose initialization copies hidden token
 *  information from the Token used to create a node.
 */
class ANTLR_API CommonASTWithHiddenTokens : public CommonAST {
public:
	CommonASTWithHiddenTokens();
	virtual ~CommonASTWithHiddenTokens();
	virtual const char* typeName( void ) const
	{
		return CommonASTWithHiddenTokens::TYPE_NAME;
	}
	/// Clone this AST node.
	virtual RefAST clone( void ) const;

	// Borland C++ builder seems to need the decl's of the first two...
	virtual void initialize(int t,const ANTLR_USE_NAMESPACE(std)string& txt);
	virtual void initialize(RefAST t);
	virtual void initialize(RefToken t);

	virtual RefToken getHiddenAfter() const
	{
		return hiddenAfter;
	}

	virtual RefToken getHiddenBefore() const
	{
		return hiddenBefore;
	}

	static RefAST factory();

	static const char* const TYPE_NAME;
protected:
	RefToken hiddenBefore,hiddenAfter; // references to hidden tokens
};

typedef ASTRefCount<CommonASTWithHiddenTokens> RefCommonASTWithHiddenTokens;

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif

#endif //INC_CommonASTWithHiddenTokens_hpp__
