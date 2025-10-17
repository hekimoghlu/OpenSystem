/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 12, 2021.
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
#include "antlr/BaseAST.hpp"
#include "antlr/CommonAST.hpp"
#include "antlr/CommonASTWithHiddenTokens.hpp"
#include "antlr/CommonHiddenStreamToken.hpp"

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

const char* const CommonASTWithHiddenTokens::TYPE_NAME = "CommonASTWithHiddenTokens";
// RK: Do not put constructor and destructor into the header file here..
// this triggers something very obscure in gcc 2.95.3 (and 3.0)
// missing vtables and stuff.
// Although this may be a problem with with binutils.
CommonASTWithHiddenTokens::CommonASTWithHiddenTokens()
: CommonAST()
{
}

CommonASTWithHiddenTokens::~CommonASTWithHiddenTokens()
{
}

void CommonASTWithHiddenTokens::initialize(int t,const ANTLR_USE_NAMESPACE(std)string& txt)
{
	CommonAST::initialize(t,txt);
}

void CommonASTWithHiddenTokens::initialize(RefAST t)
{
	CommonAST::initialize(t);
	hiddenBefore = RefCommonASTWithHiddenTokens(t)->getHiddenBefore();
	hiddenAfter = RefCommonASTWithHiddenTokens(t)->getHiddenAfter();
}

void CommonASTWithHiddenTokens::initialize(RefToken t)
{
	CommonAST::initialize(t);
	hiddenBefore = static_cast<CommonHiddenStreamToken*>(t.get())->getHiddenBefore();
	hiddenAfter = static_cast<CommonHiddenStreamToken*>(t.get())->getHiddenAfter();
}

RefAST CommonASTWithHiddenTokens::factory()
{
	return RefAST(new CommonASTWithHiddenTokens);
}

RefAST CommonASTWithHiddenTokens::clone( void ) const
{
	CommonASTWithHiddenTokens *ast = new CommonASTWithHiddenTokens( *this );
	return RefAST(ast);
}

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif
