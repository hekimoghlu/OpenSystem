/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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
#include "antlr/Token.hpp"
#include "antlr/String.hpp"

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

int Token::getColumn() const
{
	return 0;
}

int Token::getLine() const
{
	return 0;
}

ANTLR_USE_NAMESPACE(std)string Token::getText() const
{
	return "<no text>";
}

int Token::getType() const
{
	return type;
}

void Token::setColumn(int)
{
}

void Token::setLine(int)
{
}

void Token::setText(const ANTLR_USE_NAMESPACE(std)string&)
{
}

void Token::setType(int t)
{
	type = t;
}

void Token::setFilename(const ANTLR_USE_NAMESPACE(std)string&)
{
}

[[clang::no_destroy]] ANTLR_USE_NAMESPACE(std)string emptyString("");

const ANTLR_USE_NAMESPACE(std)string& Token::getFilename() const
{
	return emptyString;
}

ANTLR_USE_NAMESPACE(std)string Token::toString() const
{
	return "[\""+getText()+"\",<"+type+">]";
}

[[clang::no_destroy]] ANTLR_API RefToken nullToken;

#ifndef NO_STATIC_CONSTS
const int Token::MIN_USER_TYPE;
const int Token::NULL_TREE_LOOKAHEAD;
const int Token::INVALID_TYPE;
const int Token::EOF_TYPE;
const int Token::SKIP;
#endif

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif
