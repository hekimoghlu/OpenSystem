/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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
// DLsession.h - Framework for DL plugin modules
//
#ifndef _H_DLSESSION
#define _H_DLSESSION

#include <security_cdsa_plugin/pluginsession.h>
#include <security_cdsa_plugin/DatabaseSession.h>

namespace Security {

//
// The abstract DLPluginSession class is the common ancestor of your implementation
// object for an DL type plugin attachment session. Inherit from this and implement
// the abstract methods to define a plugin session.
//
class DLPluginSession : public PluginSession, public DatabaseSession {
    NOCOPY(DLPluginSession)
public:
    DLPluginSession(CSSM_MODULE_HANDLE theHandle,
                    CssmPlugin &plug,
                    const CSSM_VERSION &version,
                    uint32 subserviceId,
                    CSSM_SERVICE_TYPE subserviceType,
                    CSSM_ATTACH_FLAGS attachFlags,
                    const CSSM_UPCALLS &upcalls,
                    DatabaseManager &databaseManager);

	void *malloc(size_t size);
	void free(void *addr) _NOEXCEPT;
	void *realloc(void *addr, size_t size);

protected:
    CSSM_MODULE_FUNCS_PTR construct();
};

} // end namespace Security

#endif //_H_DLSESSION
