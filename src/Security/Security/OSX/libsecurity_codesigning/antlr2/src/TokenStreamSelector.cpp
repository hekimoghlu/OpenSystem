/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 17, 2024.
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
#include "antlr/TokenStreamSelector.hpp"
#include "antlr/TokenStreamRetryException.hpp"

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

/** A token stream MUX (multiplexor) knows about n token streams
 *  and can multiplex them onto the same channel for use by token
 *  stream consumer like a parser.  This is a way to have multiple
 *  lexers break up the same input stream for a single parser.
 *	Or, you can have multiple instances of the same lexer handle
 *  multiple input streams; this works great for includes.
 */

TokenStreamSelector::TokenStreamSelector()
: input(0)
{
}

TokenStreamSelector::~TokenStreamSelector()
{
}

void TokenStreamSelector::addInputStream(TokenStream* stream, const ANTLR_USE_NAMESPACE(std)string& key)
{
	inputStreamNames[key] = stream;
}

TokenStream* TokenStreamSelector::getCurrentStream() const
{
	return input;
}

TokenStream* TokenStreamSelector::getStream(const ANTLR_USE_NAMESPACE(std)string& sname) const
{
	inputStreamNames_coll::const_iterator i = inputStreamNames.find(sname);
	if (i == inputStreamNames.end()) {
		throw ANTLR_USE_NAMESPACE(std)string("TokenStream ")+sname+" not found";
	}
	return (*i).second;
}

RefToken TokenStreamSelector::nextToken()
{
	// keep looking for a token until you don't
	// get a retry exception
	for (;;) {
		try {
			return input->nextToken();
		}
		catch (TokenStreamRetryException&) {
			// just retry "forever"
		}
	}
}

TokenStream* TokenStreamSelector::pop()
{
	TokenStream* stream = streamStack.top();
	streamStack.pop();
	select(stream);
	return stream;
}

void TokenStreamSelector::push(TokenStream* stream)
{
	streamStack.push(input);
	select(stream);
}

void TokenStreamSelector::push(const ANTLR_USE_NAMESPACE(std)string& sname)
{
	streamStack.push(input);
	select(sname);
}

void TokenStreamSelector::retry()
{
	throw TokenStreamRetryException();
}

/** Set the stream without pushing old stream */
void TokenStreamSelector::select(TokenStream* stream)
{
	input = stream;
}

void TokenStreamSelector::select(const ANTLR_USE_NAMESPACE(std)string& sname)
{
	inputStreamNames_coll::const_iterator i = inputStreamNames.find(sname);
	if (i == inputStreamNames.end()) {
		throw ANTLR_USE_NAMESPACE(std)string("TokenStream ")+sname+" not found";
	}
	input = (*i).second;
}

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif

