/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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
// SDCSPDLSession.h - File Based CSP/DL plug-in module.
//
#ifndef _H_SDCSPDLSESSION
#define _H_SDCSPDLSESSION

#include <security_cdsa_plugin/CSPsession.h>
#include <securityd_client/ssclient.h>


class SDCSPDLPlugin;
class SDFactory;
class SDCSPSession;
class SDKey;

class SDCSPDLSession: public KeyPool
{
public:
	SDCSPDLSession();

	void makeReferenceKey(SDCSPSession &session,
						  SecurityServer::KeyHandle inKeyHandle,
						  CssmKey &outKey, CSSM_DB_HANDLE inDBHandle,
						  uint32 inKeyAttr, const CssmData *inKeyLabel);
	SDKey &lookupKey(const CssmKey &inKey);
};


#endif // _H_SDCSPDLSESSION
