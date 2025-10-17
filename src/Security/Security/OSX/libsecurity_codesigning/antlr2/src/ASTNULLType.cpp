/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 17, 2023.
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
#include "antlr/config.hpp"
#include "antlr/AST.hpp"
#include "antlr/ASTNULLType.hpp"

//#include <iostream>

ANTLR_USING_NAMESPACE(std)

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

RefAST ASTNULLType::clone( void ) const
{
	return RefAST(this);
}

void ASTNULLType::addChild( RefAST )
{
}

size_t ASTNULLType::getNumberOfChildren() const
{
	return 0;
}

bool ASTNULLType::equals( RefAST ) const
{
	return false;
}

bool ASTNULLType::equalsList( RefAST ) const
{
	return false;
}

bool ASTNULLType::equalsListPartial( RefAST ) const
{
	return false;
}

bool ASTNULLType::equalsTree( RefAST ) const
{
	return false;
}

bool ASTNULLType::equalsTreePartial( RefAST ) const
{
	return false;
}

vector<RefAST> ASTNULLType::findAll( RefAST )
{
	return vector<RefAST>();
}

vector<RefAST> ASTNULLType::findAllPartial( RefAST )
{
	return vector<RefAST>();
}

RefAST ASTNULLType::getFirstChild() const
{
	return this;
}

RefAST ASTNULLType::getNextSibling() const
{
	return this;
}

string ASTNULLType::getText() const
{
	return "<ASTNULL>";
}

int ASTNULLType::getType() const
{
	return Token::NULL_TREE_LOOKAHEAD;
}

void ASTNULLType::initialize( int, const string& )
{
}

void ASTNULLType::initialize( RefAST )
{
}

void ASTNULLType::initialize( RefToken )
{
}

#ifdef ANTLR_SUPPORT_XML
void ASTNULLType::initialize( istream& )
{
}
#endif

void ASTNULLType::setFirstChild( RefAST )
{
}

void ASTNULLType::setNextSibling( RefAST )
{
}

void ASTNULLType::setText( const string& )
{
}

void ASTNULLType::setType( int )
{
}

string ASTNULLType::toString() const
{
	return getText();
}

string ASTNULLType::toStringList() const
{
	return getText();
}

string ASTNULLType::toStringTree() const
{
	return getText();
}

#ifdef ANTLR_SUPPORT_XML
bool ASTNULLType::attributesToStream( ostream& ) const
{
	return false;
}

void ASTNULLType::toStream( ostream& out ) const
{
	out << "</ASTNULL>" << endl;
}
#endif

const char* ASTNULLType::typeName( void ) const
{
	return "ASTNULLType";
}

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif
