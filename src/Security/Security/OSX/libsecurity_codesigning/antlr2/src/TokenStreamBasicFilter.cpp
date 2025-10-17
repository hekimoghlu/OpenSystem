/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 13, 2024.
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
#include "antlr/TokenStreamBasicFilter.hpp"

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

/** This object is a TokenStream that passes through all
 *  tokens except for those that you tell it to discard.
 *  There is no buffering of the tokens.
 */
TokenStreamBasicFilter::TokenStreamBasicFilter(TokenStream& input_)
: input(&input_)
{
}

void TokenStreamBasicFilter::discard(int ttype)
{
	discardMask.add(ttype);
}

void TokenStreamBasicFilter::discard(const BitSet& mask)
{
	discardMask = mask;
}

RefToken TokenStreamBasicFilter::nextToken()
{
	RefToken tok = input->nextToken();
	while ( tok && discardMask.member(tok->getType()) ) {
		tok = input->nextToken();
	}
	return tok;
}

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif

