/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 9, 2025.
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
// cssmplugin - adapter framework for C++-based CDSA plugin modules
//
// A note on locking: Attachments are effectively reference counted in CSSM.
// CSSM will not let a client detach an attachment that has a(nother) thread
// active in its code. Thus, our locks merely protect global maps; they do not
// need (or try) to close the classic use-and-delete window.
//
#include <security_cdsa_plugin/cssmplugin.h>
#include <security_cdsa_plugin/pluginsession.h>
#include <memory>
#include "LegacyAPICounts.h"


ModuleNexus<CssmPlugin::SessionMap> CssmPlugin::sessionMap;


CssmPlugin::CssmPlugin()
	: mLoaded(false)
{
}

CssmPlugin::~CssmPlugin()
{
	// Note: if mLoaded, we're being unloaded forcibly.
	// (CSSM wouldn't do this to us in normal operation.)
}


//
// Load processing.
// CSSM only calls this once for a module, and multiplexes any additional
// CSSM_ModuleLoad calls internally. So this is only called when we have just
// been loaded (and not yet attached).
//
void CssmPlugin::moduleLoad(const Guid &cssmGuid,
                const Guid &moduleGuid,
                const ModuleCallback &newCallback)
{
    static dispatch_once_t onceToken;
    countLegacyAPI(&onceToken, "CssmPlugin::moduleLoad");
    if (mLoaded) {
        CssmError::throwMe(CSSM_ERRCODE_INTERNAL_ERROR);
    }

    mMyGuid = moduleGuid;

    // let the implementation know that we're loading
	this->load();

    // commit
    mCallback = newCallback;
    mLoaded = true;
}


//
// Unload processing.
// The callback passed here will be the same passed to load.
// CSSM only calls this on a "final" CSSM_ModuleUnload, after all attachments
// are destroyed and (just) before we are physically unloaded.
//
void CssmPlugin::moduleUnload(const Guid &cssmGuid,
				const Guid &moduleGuid,
                const ModuleCallback &oldCallback)
{
    // These are called from the public pluginspi.h
    static dispatch_once_t onceToken;
    countLegacyAPI(&onceToken, "CssmPlugin::moduleUnload");
    // check the callback vector
    if (!mLoaded || oldCallback != mCallback) {
        CssmError::throwMe(CSSM_ERRCODE_INTERNAL_ERROR);
    }

    // tell our subclass that we're closing down
    this->unload();

    // commit closure
    mLoaded = false;
}


//
// Create one attachment session. This is what CSSM calls to process
// a CSSM_ModuleAttach call. moduleLoad() has already been called and has
// returned successfully.
//
void CssmPlugin::moduleAttach(CSSM_MODULE_HANDLE theHandle,
                              const Guid &newCssmGuid,
                              const Guid &moduleGuid,
                              const Guid &moduleManagerGuid,
                              const Guid &callerGuid,
                              const CSSM_VERSION &version,
                              uint32 subserviceId,
                              CSSM_SERVICE_TYPE subserviceType,
                              CSSM_ATTACH_FLAGS attachFlags,
                              CSSM_KEY_HIERARCHY keyHierarchy,
                              const CSSM_UPCALLS &upcalls,
                              CSSM_MODULE_FUNCS_PTR &funcTbl)
{
	static dispatch_once_t onceToken;
	countLegacyAPI(&onceToken, "CssmPlugin::moduleAttach");
	// basic checks
	if (moduleGuid != mMyGuid)
		CssmError::throwMe(CSSM_ERRCODE_INVALID_GUID);
    
    // make the new session object, hanging in thin air
    unique_ptr<PluginSession> session(this->makeSession(theHandle,
                                         version,
                                         subserviceId, subserviceType,
                                         attachFlags,
                                         upcalls));

	// haggle with the implementor
	funcTbl = session->construct();

	// commit this session creation
    StLock<Mutex> _(sessionMap());
	sessionMap()[theHandle] = session.release();
}


//
// Undo a (single) module attachment. This calls the detach() method on
// the Session object representing the attachment. This is only called
// if session->construct() has succeeded previously.
// If session->detach() fails, we do not destroy the session and it continues
// to live, though its handle may have (briefly) been invalid. This is for
// desperate "mustn't go right now" situations and should not be abused.
// CSSM always has the ability to ditch you without your consent if you are
// obstreporous.
//
void CssmPlugin::moduleDetach(CSSM_MODULE_HANDLE handle)
{
	static dispatch_once_t onceToken;
	countLegacyAPI(&onceToken, "CssmPlugin::moduleDetach");
	// locate the plugin and hold the sessionMapLock
	PluginSession *session;
	{
		StLock<Mutex> _(sessionMap());
		SessionMap::iterator it = sessionMap().find(handle);
		if (it == sessionMap().end())
			CssmError::throwMe(CSSMERR_CSSM_INVALID_ADDIN_HANDLE);
		session = it->second;
		sessionMap().erase(it);
	}
		
	// let the session know it is going away
	try {
		session->detach();
		delete session;
	} catch (...) {
		// session detach failed - put the plugin back and fail
		StLock<Mutex> _(sessionMap());
		sessionMap()[handle] = session;
		throw;
	}
}


//
// Send an official CSSM module callback message upstream
//
void CssmPlugin::sendCallback(CSSM_MODULE_EVENT event, uint32 ssid,
                     		  CSSM_SERVICE_TYPE serviceType) const
{
	assert(mLoaded);
	mCallback(event, mMyGuid, ssid, serviceType);
}


//
// Default subclass hooks.
// The default implementations succeed without doing anything.
//
void CssmPlugin::load() { }

void CssmPlugin::unload() { }
