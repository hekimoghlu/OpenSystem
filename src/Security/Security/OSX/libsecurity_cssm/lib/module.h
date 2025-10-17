/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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
// module - CSSM Module objects
//
#ifndef _H_MODULE
#define _H_MODULE

#include "cssmint.h"
#include "cssmmds.h"
#include <security_cdsa_utilities/callback.h>
#include "modloader.h"
#include <set>

#include <unordered_map>


//
// This type represents a loaded plugin module of some kind. For each CssmManager
// instance and each one plugin, there is only (at most) one Module object to
// represent it.
//
class Module : public MdsComponent {
    NOCOPY(Module)
public:
    Module(CssmManager *mgr, const MdsComponent &info, Plugin *plugin);
    virtual ~Module();

    CssmManager &cssm;

    bool unload(const ModuleCallback &callback);

    CSSM_HANDLE attach(const CSSM_VERSION &version,
                       uint32 subserviceId,
                       CSSM_SERVICE_TYPE subserviceType,
                       const CSSM_API_MEMORY_FUNCS &memoryOps,
                       CSSM_ATTACH_FLAGS attachFlags,
                       CSSM_KEY_HIERARCHY keyHierarchy,
                       CSSM_FUNC_NAME_ADDR *functionTable,
                       uint32 functionTableSize);
    void detach(Attachment *attachment);

    void add(const ModuleCallback &cb) { callbackSet.insert(cb); }
    void remove(const ModuleCallback &cb) { callbackSet.erase(cb); }

    unsigned int callbackCount() const { return callbackSet.size(); }
    unsigned int attachmentCount() const { return (int)attachmentMap.size(); }

	void safeLock()		{ if (!isThreadSafe()) mLock.lock(); }
	void safeUnlock()	{ if (!isThreadSafe()) mLock.unlock(); }
    
public:
    typedef std::unordered_map<CSSM_HANDLE, Attachment *> AttachmentMap;
    
    Plugin *plugin;					// our loader module
	
private:
    void spiEvent(CSSM_MODULE_EVENT event,
                         const Guid &guid,
                         uint32 subserviceId,
                         CSSM_SERVICE_TYPE serviceType);

    static CSSM_RETURN spiEventRelay(const CSSM_GUID *ModuleGuid,
                                     void *Context,
                                     uint32 SubserviceId,
                                     CSSM_SERVICE_TYPE ServiceType,
                                     CSSM_MODULE_EVENT EventType);

private:
    AttachmentMap attachmentMap;	// map of all outstanding attachment handles
    ModuleCallbackSet callbackSet;	// set of registered callbacks

    Mutex mLock;
};

#endif //_H_MODULE
