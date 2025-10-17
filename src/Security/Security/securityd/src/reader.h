/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 7, 2021.
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
#ifndef _H_READER
#define _H_READER

#include "structure.h"
#include "token.h"
#include "tokencache.h"
#include <security_utilities/pcsc++.h>


//
// A Reader object represents a token (card) reader device attached to the
// system.
//
class Reader : public PerGlobal {
public:
	Reader(TokenCache &cache, const PCSC::ReaderState &state);	// PCSC managed
	Reader(TokenCache &cache, const std::string &name);			// software
	~Reader();
	
	enum Type {
		pcsc,				// represents PCSC-managed reader
		software			// software (virtual) reader,
	};
	Type type() const { return mType; }
	bool isType(Type type) const;
	
	TokenCache &cache;
	
	void kill();
	
	const string &name() const { return mName; }
	const string &printName() const { return mPrintName; }
	const PCSC::ReaderState &pcscState() const { return mState; }

	void insertToken(TokenDaemon *tokend);
	void update(const PCSC::ReaderState &state);
	void removeToken();
	
	IFDUMP(void dumpNode());
	
protected:
	
private:
	Type mType;
	string mName;			// PCSC reader name
	string mPrintName;		// human readable name of reader
	PCSC::ReaderState mState; // name field not valid (use mName)
	Token *mToken;			// token inserted here (also in references)
};


#endif //_H_READER
