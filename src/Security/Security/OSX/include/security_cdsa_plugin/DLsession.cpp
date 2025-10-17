/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 28, 2025.
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
// DLsession - Plugin framework for CSP plugin modules
//
#ifdef __MWERKS__
#define _CPP_DLSESSION
#endif

#include <security_cdsa_plugin/DLsession.h>
#include <security_cdsa_plugin/cssmplugin.h>


//
// Construct a DLPluginSession
//
DLPluginSession::DLPluginSession(CSSM_MODULE_HANDLE theHandle,
                                 CssmPlugin &plug,
                                 const CSSM_VERSION &version,
                                 uint32 subserviceId,
                                 CSSM_SERVICE_TYPE subserviceType,
                                 CSSM_ATTACH_FLAGS attachFlags,
                                 const CSSM_UPCALLS &upcalls,
                                 DatabaseManager &databaseManager)
  : PluginSession(theHandle, plug, version, subserviceId, subserviceType, attachFlags, upcalls),
    DatabaseSession (databaseManager)
{
}


//
// Implement Allocator methods from the PluginSession side
//
void *DLPluginSession::malloc(size_t size)
{ return PluginSession::malloc(size); }

void DLPluginSession::free(void *addr) _NOEXCEPT
{ return PluginSession::free(addr); }

void *DLPluginSession::realloc(void *addr, size_t size)
{ return PluginSession::realloc(addr, size); }
