/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 26, 2022.
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
#include "antlr/String.hpp"

#include <cctype>

#ifdef HAS_NOT_CSTDIO_H
#include <stdio.h>
#else
#include <cstdio>
#endif

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

ANTLR_C_USING(snprintf)

ANTLR_USE_NAMESPACE(std)string operator+( const ANTLR_USE_NAMESPACE(std)string& lhs, const int rhs )
{
	char tmp[100];
	snprintf(tmp,sizeof(tmp),"%d",rhs);
	return lhs+tmp;
}

ANTLR_USE_NAMESPACE(std)string operator+( const ANTLR_USE_NAMESPACE(std)string& lhs, size_t rhs )
{
	char tmp[100];
	snprintf(tmp,sizeof(tmp),"%lu",(unsigned long)rhs);
	return lhs+tmp;
}

/** Convert character to readable string
 */
ANTLR_USE_NAMESPACE(std)string charName(int ch)
{
	if (ch == EOF)
		return "EOF";
	else
	{
		ANTLR_USE_NAMESPACE(std)string s;

		// when you think you've seen it all.. an isprint that crashes...
		ch = ch & 0xFF;
#ifdef ANTLR_CCTYPE_NEEDS_STD
		if( ANTLR_USE_NAMESPACE(std)isprint( ch ) )
#else
		if( isprint( ch ) )
#endif
		{
			s.append("'");
			s += ch;
			s.append("'");
//			s += "'"+ch+"'";
		}
		else
		{
			s += "0x";

			unsigned int t = ch >> 4;
			if( t < 10 )
				s += t | 0x30;
			else
				s += t + 0x37;
			t = ch & 0xF;
			if( t < 10 )
				s += t | 0x30;
			else
				s += t + 0x37;
		}
		return s;
	}
}

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif

