/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 3, 2025.
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
// SDCSPDLPlugin.cpp - Securityd-based CSP/DL plug-in module.
//
#include "SDCSPDLPlugin.h"

#include "SDCSPSession.h"
#include "SDDLSession.h"
#include <securityd_client/dictionary.h>

using namespace SecurityServer;


//
// Make and break the plugin object
//
SDCSPDLPlugin::SDCSPDLPlugin()
	: EventListener(kNotificationDomainCDSA, kNotificationAllEvents),
	  mRawCsp(gGuidAppleCSP)
{
    mInitialized = true;
    EventListener::FinishedInitialization(this);
}

SDCSPDLPlugin::~SDCSPDLPlugin()
{
}


//
// Create a new plugin session, our way
//
PluginSession *
SDCSPDLPlugin::makeSession(CSSM_MODULE_HANDLE handle,
						 const CSSM_VERSION &version,
						 uint32 subserviceId,
						 CSSM_SERVICE_TYPE subserviceType,
						 CSSM_ATTACH_FLAGS attachFlags,
						 const CSSM_UPCALLS &upcalls)
{
    switch (subserviceType)
	{
        case CSSM_SERVICE_CSP:
            return new SDCSPSession(handle,
									*this,
									version,
									subserviceId,
									subserviceType,
									attachFlags,
									upcalls,
									mSDCSPDLSession,
									mRawCsp);
        case CSSM_SERVICE_DL:
            return new SDDLSession(handle,
								   *this,
								   version,
								   subserviceId,
								   subserviceType,
								   attachFlags,
								   upcalls,
								   mDatabaseManager,
								   mSDCSPDLSession);
        default:
            CssmError::throwMe(CSSMERR_CSSM_INVALID_SERVICE_MASK);
//            return 0;	// placebo
    }
}


//
// Accept callback notifications from securityd and dispatch them
// upstream through CSSM.
//
void SDCSPDLPlugin::consume(NotificationDomain domain, NotificationEvent event,
	const CssmData &data)
{
	NameValueDictionary nvd(data);
	assert(domain == kNotificationDomainCDSA);
	if (const NameValuePair *uidp = nvd.FindByName(SSUID_KEY)) {
		CssmSubserviceUid *uid = (CssmSubserviceUid *)uidp->Value().data();
		assert(uid);
		secinfo("sdcspdl", "sending callback %d upstream", event);
		sendCallback(event, n2h (uid->subserviceId()), CSSM_SERVICE_DL | CSSM_SERVICE_CSP);
	} else
		secinfo("sdcspdl", "callback event %d has no SSUID data", event);
}
