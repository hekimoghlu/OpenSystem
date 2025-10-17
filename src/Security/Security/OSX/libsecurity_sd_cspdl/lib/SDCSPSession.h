/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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
// SDDLSession.h - CSP session for security server CSP/DL.
//
#ifndef _H_SDCSPSESSION
#define _H_SDCSPSESSION

#include "SDCSPDLSession.h"

#include <securityd_client/ssclient.h>
#include <security_cdsa_client/cspclient.h>


class SDCSPDLPlugin;
class SDFactory;
class SDKey;

class SDCSPSession : public CSPFullPluginSession
{
public:
	SDCSPDLSession &mSDCSPDLSession;
	SDFactory &mSDFactory;
	CssmClient::CSP &mRawCsp;
	
	SDCSPSession(CSSM_MODULE_HANDLE handle,
				 SDCSPDLPlugin &plug,
				 const CSSM_VERSION &version,
				 uint32 subserviceId,
				 CSSM_SERVICE_TYPE subserviceType,
				 CSSM_ATTACH_FLAGS attachFlags,
				 const CSSM_UPCALLS &upcalls,
				 SDCSPDLSession &ssCSPDLSession,
				 CssmClient::CSP &rawCsp);

	SecurityServer::ClientSession &clientSession()
	{ return mClientSession; }

	CSPContext *contextCreate(CSSM_CC_HANDLE handle, const Context &context);
#if 0
	void contextUpdate(CSSM_CC_HANDLE handle, const Context &context,
					   PluginContext *ctx);
	void contextDelete(CSSM_CC_HANDLE handle, const Context &context,
					   PluginContext *ctx);
#endif

	void setupContext(CSPContext * &ctx, const Context &context,
					  bool encoding);
	
	CSSM_DB_HANDLE getDatabase(CSSM_DL_DB_HANDLE *aDLDbHandle);
	CSSM_DB_HANDLE getDatabase(const Context &context);

	void makeReferenceKey(SecurityServer::KeyHandle inKeyHandle,
						  CssmKey &outKey, CSSM_DB_HANDLE inDBHandle,
						  uint32 inKeyAttr, const CssmData *inKeyLabel);
	SDKey &lookupKey(const CssmKey &inKey);

	void WrapKey(CSSM_CC_HANDLE CCHandle,
				const Context &Context,
				const AccessCredentials &AccessCred,
				const CssmKey &Key,
				const CssmData *DescriptiveData,
				CssmKey &WrappedKey,
				CSSM_PRIVILEGE Privilege);
	void UnwrapKey(CSSM_CC_HANDLE CCHandle,
				const Context &Context,
				const CssmKey *PublicKey,
				const CssmKey &WrappedKey,
				uint32 KeyUsage,
				uint32 KeyAttr,
				const CssmData *KeyLabel,
				const CSSM_RESOURCE_CONTROL_CONTEXT *CredAndAclEntry,
				CssmKey &UnwrappedKey,
				CssmData &DescriptiveData,
				CSSM_PRIVILEGE Privilege);
	void DeriveKey(CSSM_CC_HANDLE CCHandle,
				const Context &Context,
				CssmData &Param,
				uint32 KeyUsage,
				uint32 KeyAttr,
				const CssmData *KeyLabel,
				const CSSM_RESOURCE_CONTROL_CONTEXT *CredAndAclEntry,
				CssmKey &DerivedKey);
	void GenerateKey(CSSM_CC_HANDLE ccHandle,
					const Context &context,
					uint32 keyUsage,
					uint32 keyAttr,
					const CssmData *keyLabel,
					const CSSM_RESOURCE_CONTROL_CONTEXT *credAndAclEntry,
					CssmKey &key,
					CSSM_PRIVILEGE privilege);
	void GenerateKeyPair(CSSM_CC_HANDLE ccHandle,
						const Context &context,
						uint32 publicKeyUsage,
						uint32 publicKeyAttr,
						const CssmData *publicKeyLabel,
						CssmKey &publicKey,
						uint32 privateKeyUsage,
						uint32 privateKeyAttr,
						const CssmData *privateKeyLabel,
						const CSSM_RESOURCE_CONTROL_CONTEXT *credAndAclEntry,
						CssmKey &privateKey,
						CSSM_PRIVILEGE privilege);
	void ObtainPrivateKeyFromPublicKey(const CssmKey &PublicKey,
									CssmKey &PrivateKey);
	void QueryKeySizeInBits(CSSM_CC_HANDLE CCHandle,
							const Context *Context,
							const CssmKey *Key,
							CSSM_KEY_SIZE &KeySize);
	void FreeKey(const AccessCredentials *AccessCred,
				CssmKey &key, CSSM_BOOL Delete);
	void GenerateRandom(CSSM_CC_HANDLE ccHandle,
						const Context &context,
						CssmData &randomNumber);
	void Login(const AccessCredentials &AccessCred,
			const CssmData *LoginName,
			const void *Reserved);
	void Logout();
	void VerifyDevice(const CssmData &DeviceCert);
	void GetOperationalStatistics(CSPOperationalStatistics &statistics);
	void RetrieveCounter(CssmData &Counter);
	void RetrieveUniqueId(CssmData &UniqueID);
	void GetTimeValue(CSSM_ALGORITHMS TimeAlgorithm, CssmData &TimeData);
	void GetKeyOwner(const CssmKey &Key,
					CSSM_ACL_OWNER_PROTOTYPE &Owner);
	void ChangeKeyOwner(const AccessCredentials &AccessCred,
						const CssmKey &Key,
						const CSSM_ACL_OWNER_PROTOTYPE &NewOwner);
	void GetKeyAcl(const CssmKey &Key,
					const CSSM_STRING *SelectionTag,
					uint32 &NumberOfAclInfos,
					CSSM_ACL_ENTRY_INFO_PTR &AclInfos);
	void ChangeKeyAcl(const AccessCredentials &AccessCred,
					const CSSM_ACL_EDIT &AclEdit,
					const CssmKey &Key);
	void GetLoginOwner(CSSM_ACL_OWNER_PROTOTYPE &Owner);
	void ChangeLoginOwner(const AccessCredentials &AccessCred,
						const CSSM_ACL_OWNER_PROTOTYPE &NewOwner);
	void GetLoginAcl(const CSSM_STRING *SelectionTag,
					uint32 &NumberOfAclInfos,
					CSSM_ACL_ENTRY_INFO_PTR &AclInfos);
	void ChangeLoginAcl(const AccessCredentials &AccessCred,
						const CSSM_ACL_EDIT &AclEdit);
	void PassThrough(CSSM_CC_HANDLE CCHandle,
					const Context &Context,
					uint32 PassThroughId,
					const void *InData,
					void **OutData);
private:
	/* Validate requested key attr flags for newly generated keys */
	void validateKeyAttr(uint32 reqKeyAttr);

	SecurityServer::ClientSession mClientSession;
};


#endif // _H_SDCSPSESSION
