/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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

#ifndef INC_ASTNULLType_hpp__
#define INC_ASTNULLType_hpp__

/* ANTLR Translator Generator
 * Project led by Terence Parr at http://www.jGuru.com
 * Software rights: http://www.antlr.org/license.html
 *
 * $Id: //depot/code/org.antlr/release/antlr-2.7.7/lib/cpp/antlr/ASTNULLType.hpp#2 $
 */

#include <antlr/config.hpp>
#include <antlr/AST.hpp>
#include <vector>
#include <string>
#include <istream>
#include <ostream>

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

/** There is only one instance of this class **/
class ANTLR_API ASTNULLType : public AST {
public:
	const char* typeName( void ) const;
	RefAST clone( void ) const;

	void addChild(RefAST c);
	size_t getNumberOfChildren() const;
	void setFirstChild(RefAST c);
	void setNextSibling(RefAST n);

	bool equals(RefAST t) const;
	bool equalsList(RefAST t) const;
	bool equalsListPartial(RefAST t) const;
	bool equalsTree(RefAST t) const;
	bool equalsTreePartial(RefAST t) const;

	ANTLR_USE_NAMESPACE(std)vector<RefAST> findAll(RefAST tree);
	ANTLR_USE_NAMESPACE(std)vector<RefAST> findAllPartial(RefAST subtree);

	RefAST getFirstChild() const;
	RefAST getNextSibling() const;

	ANTLR_USE_NAMESPACE(std)string getText() const;
	int getType() const;

	void initialize(int t, const ANTLR_USE_NAMESPACE(std)string& txt);
	void initialize(RefAST t);
	void initialize(RefToken t);
	void initialize(ANTLR_USE_NAMESPACE(std)istream& infile);

	void setText(const ANTLR_USE_NAMESPACE(std)string& text);
	void setType(int ttype);
	ANTLR_USE_NAMESPACE(std)string toString() const;
	ANTLR_USE_NAMESPACE(std)string toStringList() const;
	ANTLR_USE_NAMESPACE(std)string toStringTree() const;

	bool attributesToStream( ANTLR_USE_NAMESPACE(std)ostream &out ) const;
	void toStream( ANTLR_USE_NAMESPACE(std)ostream &out ) const;
};

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif

#endif //INC_ASTNULLType_hpp__
