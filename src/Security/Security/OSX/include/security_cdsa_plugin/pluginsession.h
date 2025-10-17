/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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
// pluginsession - an attachment session for a CSSM plugin
//
#ifndef _H_PLUGINSESSION
#define _H_PLUGINSESSION

#include <security_cdsa_plugin/c++plugin.h>
#include <security_cdsa_utilities/handleobject.h>
#include <security_utilities/alloc.h>


namespace Security {


//
// A PluginSession object describes an ongoing connection between a particular
// CSSM client and our plugin. Every time CSSM_SPI_ModuleAttach is called
// (due to the client calling CSSM_ModuleAttach), a new PluginSession object
// is created as a result. Sessions and CSSM_MODULE_HANDLES correspond one-to-one.
// Note that CSSM makes up our module handle; we just record it.
//
// A PluginSession *is* an Allocator, whose implementation is to call the
// "application allocator" functions provided by CSSM's caller for the attachment.
// Use the session object as the Allocator for anything you return to your caller.
//
class PluginSession : public Allocator, public HandledObject {
    NOCOPY(PluginSession)
    friend class CssmPlugin;
public:
    PluginSession(CSSM_MODULE_HANDLE theHandle,
                  CssmPlugin &myPlugin,
                  const CSSM_VERSION &Version,
                  uint32 SubserviceID,
                  CSSM_SERVICE_TYPE SubServiceType,
                  CSSM_ATTACH_FLAGS AttachFlags,
                  const CSSM_UPCALLS &upcalls);
    virtual ~PluginSession();
    virtual void detach();

    CssmPlugin &plugin;
    
    void sendCallback(CSSM_MODULE_EVENT event,
    				  uint32 ssid = uint32(-1),
                      CSSM_SERVICE_TYPE serviceType = 0) const;

    static void unimplemented() { CssmError::throwMe(CSSM_ERRCODE_FUNCTION_NOT_IMPLEMENTED); }

protected:
    virtual CSSM_MODULE_FUNCS_PTR construct() = 0;

public:
    // implement Allocator
    void *malloc(size_t size);
    void *realloc(void *addr, size_t size);
    void free(void *addr) _NOEXCEPT { upcalls.free_func(handle(), addr); }

	// about ourselves
	const CSSM_VERSION &version() const { return mVersion; }
    uint32 subserviceId() const { return mSubserviceId; }
    CSSM_SERVICE_TYPE subserviceType() const { return mSubserviceType; }
    CSSM_ATTACH_FLAGS attachFlags() const { return mAttachFlags; }

private:
    CSSM_VERSION mVersion;
    uint32 mSubserviceId;
    CSSM_SERVICE_TYPE mSubserviceType;
    CSSM_ATTACH_FLAGS mAttachFlags;
    const CSSM_UPCALLS &upcalls;
};

} // end namespace Security


#endif //_H_PLUGINSESSION
