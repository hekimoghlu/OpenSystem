/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 25, 2022.
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
#ifndef _H_TOKENKEY
#define _H_TOKENKEY


//
// tokenkey - remote reference key on an attached hardware token
//
#include "key.h"
#include "tokenacl.h"

class TokenDatabase;


//
// The token-specific instance of a Key
//
class TokenKey : public Key, public TokenAcl {
public:
	TokenKey(TokenDatabase &db, KeyHandle hKey, const CssmKey::Header &hdr);
	~TokenKey();
	
	TokenDatabase &database() const;
	Token &token();
	const CssmKey::Header &header() const { return mHeader; }
	KeyHandle tokenHandle() const;
	
	CSSM_KEYATTR_FLAGS attributes();
	void returnKey(Handle &h, CssmKey::Header &hdr);
	const CssmData &canonicalDigest();
    virtual void publicKey(const Context &context, CssmData &pubKeyData);

	SecurityServerAcl &acl();
	Database *relatedDatabase();

public:
	// SecurityServerAcl personality
	AclKind aclKind() const;

private:
	KeyHandle mKey;			// tokend reference handle
	CssmKey::Header mHeader; // key header as maintained by tokend
};

#endif //_H_TOKENKEY
