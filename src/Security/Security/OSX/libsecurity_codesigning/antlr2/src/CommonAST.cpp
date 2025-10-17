/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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

#include <cstdlib>
//#include <iostream>

#include "antlr/CommonAST.hpp"
#include "antlr/ANTLRUtil.hpp"

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

const char* const CommonAST::TYPE_NAME = "CommonAST";

#ifdef ANTLR_SUPPORT_XML
void CommonAST::initialize( ANTLR_USE_NAMESPACE(std)istream& in )
{
	ANTLR_USE_NAMESPACE(std)string t1, t2, text;

	// text
	read_AttributeNValue( in, t1, text );

	read_AttributeNValue( in, t1, t2 );
#ifdef ANTLR_ATOI_IN_STD
	int type = ANTLR_USE_NAMESPACE(std)atoi(t2.c_str());
#else
	int type = atoi(t2.c_str());
#endif

	// initialize first part of AST.
	this->initialize( type, text );
}
#endif

RefAST CommonAST::factory()
{
	return RefAST(new CommonAST);
}

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif

