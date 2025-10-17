/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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
// SSKey.h - CSP-wide SSKey base class
//
#ifndef	_H_SSKEY_
#define _H_SSKEY_

#include <security_cdsa_plugin/CSPsession.h>

#include "SSDatabase.h"

#include <security_cdsa_client/dlclient.h>
#include <securityd_client/ssclient.h>

namespace Security
{

class CssmKey;

} // end namespace Security

class SSCSPSession;
class SSCSPDLSession;
class SSDLSession;

class SSKey : public ReferencedKey
{
public:
	SSKey(SSCSPSession &session, SecurityServer::KeyHandle keyHandle,
		  CssmKey &ioKey, SSDatabase &inSSDatabase, uint32 inKeyAttr,
		  const CssmData *inKeyLabel);
	SSKey(SSDLSession &session, CssmKey &ioKey, SSDatabase &inSSDatabase,
		  const SSUniqueRecord &uniqueId, CSSM_DB_RECORDTYPE recordType,
		  CssmData &keyBlob);

	virtual ~SSKey();
	void free(const AccessCredentials *accessCred, CssmKey &ioKey,
			  CSSM_BOOL deleteKey);

	SecurityServer::ClientSession &clientSession();

	/* Might return SecurityServer::noKey if the key has not yet been instantiated. */
	SecurityServer::KeyHandle optionalKeyHandle() const;

	/* Will instantiate the key if needed. */
	SecurityServer::KeyHandle keyHandle();

    // ACL retrieval and change operations
	void getOwner(CSSM_ACL_OWNER_PROTOTYPE &owner, Allocator &allocator);
	void changeOwner(const AccessCredentials &accessCred,
					 const AclOwnerPrototype &newOwner);
	void getAcl(const char *selectionTag, uint32 &numberOfAclInfos,
				AclEntryInfo *&aclInfos, Allocator &allocator);
	void changeAcl(const AccessCredentials &accessCred,
				   const AclEdit &aclEdit);

	// Reencode and write to disk if we are a persistent key.
	void didChangeAcl();

private:
	Allocator &mAllocator;
	SecurityServer::KeyHandle mKeyHandle;
	SSDatabase mSSDatabase;
	SSUniqueRecord mUniqueId;
	CSSM_DB_RECORDTYPE mRecordType;
	SecurityServer::ClientSession &mClientSession;
    mutable RecursiveMutex mMutex;
};


#endif	// _H_SSKEY_
