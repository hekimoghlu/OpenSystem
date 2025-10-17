/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 19, 2025.
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
//
// reader - token reader objects
//
#include "reader.h"


//
// Construct a Reader
// This does not commence state tracking; call update to start up the reader.
//
Reader::Reader(TokenCache &tc, const PCSC::ReaderState &state)
	: cache(tc), mType(pcsc), mToken(NULL)
{
	mName = state.name();	// remember separate copy of name
	mPrintName = mName;		//@@@ how to make this readable? Use IOKit information?
	secinfo("reader", "%p (%s) new PCSC reader", this, name().c_str());
}

Reader::Reader(TokenCache &tc, const string &identifier)
	: cache(tc), mType(software), mToken(NULL)
{
	mName = identifier;
	mPrintName = mName;
	secinfo("reader", "%p (%s) new software reader", this, name().c_str());
}

Reader::~Reader()
{
	secinfo("reader", "%p (%s) destroyed", this, name().c_str());
}


//
// Type qualification. None matches anything.
//
bool Reader::isType(Type reqType) const
{
	return reqType == this->type();
}


//
// Killing a reader forcibly removes its Token, if any
//
void Reader::kill()
{
	if (mToken)
		removeToken();
	NodeCore::kill();
}


//
// State transition matrix for a reader, based on PCSC state changes
//
void Reader::update(const PCSC::ReaderState &state)
{
	// set new state
	unsigned long oldState = mState.state();
    (void) oldState; // Be okay with not using this.

	mState = state;
	mState.name(mName.c_str());		// (fix name pointer, unchanged)
	
	try {
		if (state.state(SCARD_STATE_UNAVAILABLE)) {
			// reader is unusable (probably being removed)
			secinfo("reader", "%p (%s) unavailable (0x%lx)",
				this, name().c_str(), state.state());
			if (mToken)
				removeToken();
		} else if (state.state(SCARD_STATE_EMPTY)) {
			// reader is empty (no token present)
			secinfo("reader", "%p (%s) empty (0x%lx)",
				this, name().c_str(), state.state());
			if (mToken)
				removeToken();
		} else if (state.state(SCARD_STATE_PRESENT)) {
			// reader has a token inserted
			secinfo("reader", "%p (%s) card present (0x%lx)",
				this, name().c_str(), state.state());
			//@@@ is this hack worth it (with notifications in)??
			if (mToken && CssmData(state) != CssmData(pcscState()))
				removeToken();  // incomplete but better than nothing
			//@@@ or should we call some verify-still-the-same function of tokend?
			//@@@ (I think pcsc will return an error if the card changed?)
			if (!mToken)
				insertToken(NULL);
		} else {
			secinfo("reader", "%p (%s) unexpected state change (0x%lx to 0x%lx)",
				this, name().c_str(), oldState, state.state());
		}
	} catch (...) {
		secinfo("reader", "state update exception (ignored)");
	}
}


void Reader::insertToken(TokenDaemon *tokend)
{
	RefPointer<Token> token = new Token();
	token->insert(*this, tokend);
	mToken = token;
	addReference(*token);
	secinfo("reader", "%p (%s) inserted token %p",
		this, name().c_str(), mToken);
}


void Reader::removeToken()
{
	secinfo("reader", "%p (%s) removing token %p",
		this, name().c_str(), mToken);
	assert(mToken);
	mToken->remove();
	removeReference(*mToken);
	mToken = NULL;
}


//
// Debug dump support
//
#if defined(DEBUGDUMP)

void Reader::dumpNode()
{
	PerGlobal::dumpNode();
	Debug::dump(" [%s] state=0x%lx", name().c_str(), mState.state());
}

#endif //DEBUGDUMP
