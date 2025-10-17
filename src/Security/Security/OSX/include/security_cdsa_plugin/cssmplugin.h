/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 13, 2025.
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
// cssmplugin - common header for CSSM plugin modules
//
#ifndef _H_CSSMPLUGIN
#define _H_CSSMPLUGIN

#include <security_cdsa_plugin/c++plugin.h>
#include <security_utilities/globalizer.h>
#include <security_cdsa_utilities/callback.h>
#include <set>

#include <unordered_map>

namespace Security {


//
// Inherit from this (abstract) class to implement your plugin
//
class CssmPlugin {
    NOCOPY(CssmPlugin)
public:
    CssmPlugin();
    virtual ~CssmPlugin();

    void moduleLoad(const Guid &cssmGuid,
                    const Guid &moduleGuid,
                    const ModuleCallback &callback);
    void moduleUnload(const Guid &cssmGuid,
                      const Guid &moduleGuid,
                      const ModuleCallback &callback);

    void moduleAttach(CSSM_MODULE_HANDLE theHandle,
                      const Guid &cssmGuid,
                      const Guid &moduleGuid,
                      const Guid &moduleManagerGuid,
                      const Guid &callerGuid,
                      const CSSM_VERSION &Version,
                      uint32 SubserviceID,
                      CSSM_SERVICE_TYPE SubServiceType,
                      CSSM_ATTACH_FLAGS AttachFlags,
                      CSSM_KEY_HIERARCHY KeyHierarchy,
                      const CSSM_UPCALLS &Upcalls,
                      CSSM_MODULE_FUNCS_PTR &FuncTbl);
    void moduleDetach(CSSM_MODULE_HANDLE handle);
    
    const Guid &myGuid() const { return mMyGuid; }
    
    void sendCallback(CSSM_MODULE_EVENT event,
                      uint32 ssid,
                      CSSM_SERVICE_TYPE serviceType) const;
                      
    void sendInsertion(uint32 subId, CSSM_SERVICE_TYPE serviceType) const
    { sendCallback(CSSM_NOTIFY_INSERT, subId, serviceType); }
                      
    void sendRemoval(uint32 subId, CSSM_SERVICE_TYPE serviceType) const
    { sendCallback(CSSM_NOTIFY_REMOVE, subId, serviceType); }
                      
    void sendFault(uint32 subId, CSSM_SERVICE_TYPE serviceType) const
    { sendCallback(CSSM_NOTIFY_FAULT, subId, serviceType); }

protected:
    // subclass-defined methods
    virtual void load();
    virtual void unload();

    // make a session object for your plugin
    virtual PluginSession *makeSession(CSSM_MODULE_HANDLE handle,
                                       const CSSM_VERSION &version,
                                       uint32 subserviceId,
                                       CSSM_SERVICE_TYPE subserviceType,
                                       CSSM_ATTACH_FLAGS attachFlags,
                                       const CSSM_UPCALLS &upcalls) = 0;

private:
    // map of (CSSM) handles to attachment objects
    struct SessionMap :
            public std::unordered_map<CSSM_MODULE_HANDLE, PluginSession *>,
            public Mutex { };
            
    static ModuleNexus<SessionMap> sessionMap;
    
    Guid mMyGuid;

    // the registered callback. Set during load processing, unset during unload
    ModuleCallback mCallback;
    bool mLoaded;

public:
    static PluginSession *find(CSSM_MODULE_HANDLE h)
    {
        StLock<Mutex> _(sessionMap());
        SessionMap::iterator it = sessionMap().find(h);
        if (it == sessionMap().end())
            CssmError::throwMe(CSSMERR_CSSM_INVALID_ADDIN_HANDLE);
        return it->second;
    }
};

template <class SessionClass>
inline SessionClass &findSession(CSSM_MODULE_HANDLE h)
{
    SessionClass *session = dynamic_cast<SessionClass *>(CssmPlugin::find(h));
    if (session == NULL)
        CssmError::throwMe(CSSMERR_CSSM_INVALID_ADDIN_HANDLE);
    assert(session->handle() == h);
    return *session;
}

} // end namespace Security

#endif //_H_CSSMPLUGIN
