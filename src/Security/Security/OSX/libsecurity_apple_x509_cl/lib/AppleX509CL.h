/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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
// AppleX509CL.h - File Based CSP/DL plug-in module.
//
#ifndef _H_APPLEX509CL
#define _H_APPLEX509CL

#include <security_cdsa_plugin/cssmplugin.h>
#include <security_cdsa_plugin/pluginsession.h>

class AppleX509CL : public CssmPlugin
{
public:
    AppleX509CL();
    ~AppleX509CL();

    PluginSession *makeSession(
			CSSM_MODULE_HANDLE handle,
			const CSSM_VERSION &version,
			uint32 subserviceId,
			CSSM_SERVICE_TYPE subserviceType,
			CSSM_ATTACH_FLAGS attachFlags,
			const CSSM_UPCALLS &upcalls);
private:
    // Don't copy AppleX509CL
    AppleX509CL(const AppleX509CL&);
    void operator=(const AppleX509CL&);
};


#endif //_H_APPLEX509CL
