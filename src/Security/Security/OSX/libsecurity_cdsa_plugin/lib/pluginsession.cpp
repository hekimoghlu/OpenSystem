/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 7, 2022.
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
#include <security_cdsa_plugin/pluginsession.h>
#include <security_cdsa_plugin/cssmplugin.h>
#include <security_cdsa_plugin/DLsession.h>


//
// Construct a PluginSession
//
PluginSession::PluginSession(CSSM_MODULE_HANDLE theHandle,
                             CssmPlugin &myPlugin,
                             const CSSM_VERSION &version,
                             uint32 subserviceId,
                             CSSM_SERVICE_TYPE subserviceType,
                             CSSM_ATTACH_FLAGS attachFlags,
                             const CSSM_UPCALLS &inUpcalls)
	: HandledObject(theHandle), plugin(myPlugin),
	  mVersion(version), mSubserviceId(subserviceId),
	  mSubserviceType(subserviceType), mAttachFlags(attachFlags),
	  upcalls(inUpcalls)
{
}


//
// Destruction
//
PluginSession::~PluginSession()
{
}


//
// The default implementation of detach() does nothing
//
void PluginSession::detach()
{
}


//
// Allocation management
//
void *PluginSession::malloc(size_t size)
{
    if (void *addr = upcalls.malloc_func(handle(), size)) {
        return addr;
    }
    throw std::bad_alloc();
}

void *PluginSession::realloc(void *oldAddr, size_t size)
{
    if (void *addr = upcalls.realloc_func(handle(), oldAddr, size)) {
        return addr;
    }
    throw std::bad_alloc();
}


//
// Dispatch callback events through the plugin object.
// Subsystem ID and subservice type default to our own.
//

void PluginSession::sendCallback(CSSM_MODULE_EVENT event,
                                 uint32 ssid,
                                 CSSM_SERVICE_TYPE serviceType) const
{
    plugin.sendCallback(event,
                        (ssid == uint32(-1)) ? mSubserviceId : ssid,
                        serviceType ? serviceType : mSubserviceType);
}
