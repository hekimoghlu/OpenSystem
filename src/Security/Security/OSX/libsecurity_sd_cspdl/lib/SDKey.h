/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 7, 2024.
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
// SDKey.h - CSP-wide SDKey base class
//
#ifndef	_H_SDKEY_
#define _H_SDKEY_

#include <securityd_client/ssclient.h>
#include <security_cdsa_plugin/CSPsession.h>

namespace Security
{

class CssmKey;

} // end namespace Security

class SDCSPSession;
class SDCSPDLSession;
class SDDLSession;

class SDKey : public ReferencedKey
{
public:
	SDKey(SDCSPSession &session, SecurityServer::KeyHandle keyHandle,
		  CssmKey &ioKey, CSSM_DB_HANDLE inDBHandle, uint32 inKeyAttr,
		  const CssmData *inKeyLabel);
	SDKey(SDDLSession &session, CssmKey &ioKey, SecurityServer::KeyHandle hKey, CSSM_DB_HANDLE inDBHandle,
		  SecurityServer::RecordHandle record, CSSM_DB_RECORDTYPE recordType,
		  CssmData &keyBlob);
	
	virtual ~SDKey();
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

private:
	Allocator &mAllocator;
	SecurityServer::KeyHandle mKeyHandle;
	CSSM_DB_HANDLE mDatabase;
	SecurityServer::RecordHandle mRecord;
	SecurityServer::ClientSession &mClientSession;
};


#endif	// _H_SDKEY_
