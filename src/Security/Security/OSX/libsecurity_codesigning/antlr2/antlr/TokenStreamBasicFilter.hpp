/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 29, 2024.
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

#ifndef INC_TokenStreamBasicFilter_hpp__
#define INC_TokenStreamBasicFilter_hpp__

/* ANTLR Translator Generator
 * Project led by Terence Parr at http://www.jGuru.com
 * Software rights: http://www.antlr.org/license.html
 *
 * $Id: //depot/code/org.antlr/release/antlr-2.7.7/lib/cpp/antlr/TokenStreamBasicFilter.hpp#2 $
 */

#include <antlr/config.hpp>
#include <antlr/BitSet.hpp>
#include <antlr/TokenStream.hpp>

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

/** This object is a TokenStream that passes through all
 *  tokens except for those that you tell it to discard.
 *  There is no buffering of the tokens.
 */
class ANTLR_API TokenStreamBasicFilter : public TokenStream {
	/** The set of token types to discard */
protected:
	BitSet discardMask;

	/** The input stream */
protected:
	TokenStream* input;

public:
	TokenStreamBasicFilter(TokenStream& input_);

	void discard(int ttype);

	void discard(const BitSet& mask);

	RefToken nextToken();
};

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif

#endif //INC_TokenStreamBasicFilter_hpp__
