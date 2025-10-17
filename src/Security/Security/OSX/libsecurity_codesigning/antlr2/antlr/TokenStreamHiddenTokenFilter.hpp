/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 19, 2022.
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

#ifndef INC_TokenStreamHiddenTokenFilter_hpp__
#define INC_TokenStreamHiddenTokenFilter_hpp__

/* ANTLR Translator Generator
 * Project led by Terence Parr at http://www.jGuru.com
 * Software rights: http://www.antlr.org/license.html
 *
 * $Id: //depot/code/org.antlr/release/antlr-2.7.7/lib/cpp/antlr/TokenStreamHiddenTokenFilter.hpp#2 $
 */

#include <antlr/config.hpp>
#include <antlr/TokenStreamBasicFilter.hpp>

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

/**This object filters a token stream coming from a lexer
 * or another TokenStream so that only certain token channels
 * get transmitted to the parser.
 *
 * Any of the channels can be filtered off as "hidden" channels whose
 * tokens can be accessed from the parser.
 */
class ANTLR_API TokenStreamHiddenTokenFilter : public TokenStreamBasicFilter {
	// protected BitSet discardMask;
protected:
	BitSet hideMask;

private:
	RefToken nextMonitoredToken;

protected:
	/** track tail of hidden list emanating from previous
	 *  monitored token
	 */
	RefToken lastHiddenToken;

	RefToken firstHidden; // = null;

public:
	TokenStreamHiddenTokenFilter(TokenStream& input);

protected:
	void consume();

private:
	void consumeFirst();

public:
	BitSet getDiscardMask() const;

	/** Return a ptr to the hidden token appearing immediately after
	 *  token t in the input stream.
	 */
	RefToken getHiddenAfter(RefToken t);

	/** Return a ptr to the hidden token appearing immediately before
	 *  token t in the input stream.
	 */
	RefToken getHiddenBefore(RefToken t);

	BitSet getHideMask() const;

	/** Return the first hidden token if one appears
	 *  before any monitored token.
	 */
	RefToken getInitialHiddenToken();

	void hide(int m);

	void hide(const BitSet& mask);

protected:
	RefToken LA(int i);

public:
/** Return the next monitored token.
 *  Test the token following the monitored token.
 *  If following is another monitored token, save it
 *  for the next invocation of nextToken (like a single
 *  lookahead token) and return it then.
 *  If following is unmonitored, nondiscarded (hidden)
 *  channel token, add it to the monitored token.
 *
 *  Note: EOF must be a monitored Token.
 */
	RefToken nextToken();
};

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif

#endif //INC_TokenStreamHiddenTokenFilter_hpp__
