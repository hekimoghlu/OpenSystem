/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 2, 2025.
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
// key - representation of SecurityServer key objects
//
#ifndef _H_KCKEY
#define _H_KCKEY

#include "localkey.h"
#include <security_cdsa_utilities/handleobject.h>
#include <security_cdsa_client/keyclient.h>


class KeychainDatabase;


//
// A KeychainKey object represents a CssmKey that is stored in a KeychainDatabase.
//
// This is a LocalKey with deferred instantiation. A KeychainKey always exists in one of
// two states:
//  (*) Decoded: The CssmKey is valid; the blob may or may not be.
//  (*) Encoded: The blob is valid, the CssmKey may or may not be.
// One of (blob, CssmKey) is always valid. The process of decoding the CssmKey from the
// blob (and vice versa) requires keychain cryptography, which unlocks the keychain
// (implicitly as needed).
// Other than that, this is just a LocalKey.
//
class KeychainKey : public LocalKey, public SecurityServerAcl {
public:
	KeychainKey(Database &db, const KeyBlob *blob);
	KeychainKey(Database &db, const CssmKey &newKey, uint32 moreAttributes,
		const AclEntryPrototype *owner = NULL);
	virtual ~KeychainKey();
    
	KeychainDatabase &database() const;
    
    // we can also yield an encoded KeyBlob
	KeyBlob *blob();
	
	void invalidateBlob();
    
    // ACL state management hooks
	void instantiateAcl();
	void changedAcl();
    Database *relatedDatabase();
	void validate(AclAuthorization auth, const AccessCredentials *cred, Database *relatedDatabase);

public:
	// SecurityServerAcl personality
	AclKind aclKind() const;
	
	SecurityServerAcl &acl();
	
private:
    void decode();
	void getKey();
	virtual void getHeader(CssmKey::Header &hdr); // get header (only) without mKey

private:
	KeyBlob *mBlob;			// key blob encoded by mDatabase
	bool mValidBlob;		// mBlob is valid key encoding
};


#endif //_H_KCKEY
