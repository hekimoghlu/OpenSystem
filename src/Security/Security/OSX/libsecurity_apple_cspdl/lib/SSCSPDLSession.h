/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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
// SSCSPDLSession.h - File Based CSP/DL plug-in module.
//
#ifndef _H_SSCSPDLSESSION
#define _H_SSCSPDLSESSION

#include <security_cdsa_plugin/CSPsession.h>
#include <securityd_client/ssclient.h>


class CSPDLPlugin;
class SSFactory;
class SSCSPSession;
class SSDatabase;
class SSKey;

class SSCSPDLSession: public KeyPool
{
public:
	SSCSPDLSession();

	void makeReferenceKey(SSCSPSession &session,
						  SecurityServer::KeyHandle inKeyHandle,
						  CssmKey &outKey, SSDatabase &inSSDatabase,
						  uint32 inKeyAttr, const CssmData *inKeyLabel);
	SSKey &lookupKey(const CssmKey &inKey);

	/* Notification we receive when a key's acl has been modified. */
	void didChangeKeyAcl(SecurityServer::ClientSession &clientSession,
		SecurityServer::KeyHandle keyHandle, CSSM_ACL_AUTHORIZATION_TAG tag);

	static void didChangeKeyAclCallback(void *context, SecurityServer::ClientSession &clientSession,
		SecurityServer::KeyHandle keyHandle, CSSM_ACL_AUTHORIZATION_TAG tag);

    Mutex mKeyDeletionMutex; // Used to ensure that only one thread is in either SSCSPSession::FreeKey or SSCSPDLSession::didChangeKeyAcl at a time, since SSCSPDLSession::didChangeKeyAcl might be trying to use the free-ing key (from a securityd callback).
};


#endif // _H_SSCSPDLSESSION
