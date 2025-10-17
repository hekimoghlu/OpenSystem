/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 29, 2023.
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
// tokenkey - remote reference key on an attached hardware token
//
#include "tokenkey.h"
#include "tokendatabase.h"


//
// Construct a TokenKey from a reference handle and key header
//
TokenKey::TokenKey(TokenDatabase &db, KeyHandle tokenKey, const CssmKey::Header &hdr)
	: Key(db), mKey(tokenKey), mHeader(hdr)
{
	db.addReference(*this);
}


//
// Destruction of a TokenKey releases the reference from tokend
//
TokenKey::~TokenKey()
{
	try {
		database().token().tokend().releaseKey(mKey);
	} catch (...) {
		secinfo("tokendb", "%p release key handle %u threw (ignored)",
			this, mKey);
	}
}


//
// Links through the object mesh
//
TokenDatabase &TokenKey::database() const
{
	return referent<TokenDatabase>();
}

Token &TokenKey::token()
{
	return database().token();
}

GenericHandle TokenKey::tokenHandle() const
{
	return mKey;	// tokend-side handle
}


//
// Canonical external attributes (taken directly from the key header)
//
CSSM_KEYATTR_FLAGS TokenKey::attributes()
{
	return mHeader.attributes();
}


//
// Return-to-caller processing (trivial in this case)
//
void TokenKey::returnKey(Handle &h, CssmKey::Header &hdr)
{
	h = this->handle();
	hdr = mHeader;
}


//
// We're a key (duh)
//
AclKind TokenKey::aclKind() const
{
	return keyAcl;
}


//
// Right now, key ACLs are at the process level
//
SecurityServerAcl &TokenKey::acl()
{
	return *this;
}


//
// The related database is, naturally enough, the TokenDatabase we're in
//
Database *TokenKey::relatedDatabase()
{
	return &database();
}


//
// Generate the canonical key digest.
// This is not currently supported through tokend. If we need it,
// we'll have to force unlock and fake it (in tokend, most likely).
//
const CssmData &TokenKey::canonicalDigest()
{
	CssmError::throwMe(CSSM_ERRCODE_FUNCTION_NOT_IMPLEMENTED);
}

void TokenKey::publicKey(const Context &context, CssmData &pubKeyData)
{
    CssmError::throwMe(CSSM_ERRCODE_FUNCTION_NOT_IMPLEMENTED);
}
