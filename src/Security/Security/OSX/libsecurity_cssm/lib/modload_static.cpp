/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 6, 2025.
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
// modload_static - pseudo-loading of statically linked plugins
//
#include "modload_static.h"


namespace Security {


//
// Pass module entry points to the statically linked functions
//
CSSM_RETURN StaticPlugin::load(const CSSM_GUID *CssmGuid,
                             const CSSM_GUID *ModuleGuid,
                             CSSM_SPI_ModuleEventHandler CssmNotifyCallback,
                             void *CssmNotifyCallbackCtx)
{
	return entries.load(CssmGuid, ModuleGuid,
		CssmNotifyCallback, CssmNotifyCallbackCtx);
}

CSSM_RETURN StaticPlugin::unload(const CSSM_GUID *CssmGuid,
                             const CSSM_GUID *ModuleGuid,
                             CSSM_SPI_ModuleEventHandler CssmNotifyCallback,
                             void *CssmNotifyCallbackCtx)
{
	return entries.unload(CssmGuid, ModuleGuid,
		CssmNotifyCallback, CssmNotifyCallbackCtx);
}

CSSM_RETURN StaticPlugin::attach(const CSSM_GUID *ModuleGuid,
                               const CSSM_VERSION *Version,
                               uint32 SubserviceID,
                               CSSM_SERVICE_TYPE SubServiceType,
                               CSSM_ATTACH_FLAGS AttachFlags,
                               CSSM_MODULE_HANDLE ModuleHandle,
                               CSSM_KEY_HIERARCHY KeyHierarchy,
                               const CSSM_GUID *CssmGuid,
                               const CSSM_GUID *ModuleManagerGuid,
                               const CSSM_GUID *CallerGuid,
                               const CSSM_UPCALLS *Upcalls,
                               CSSM_MODULE_FUNCS_PTR *FuncTbl)
{
	return entries.attach(ModuleGuid, Version, SubserviceID, SubServiceType,
		AttachFlags, ModuleHandle, KeyHierarchy, CssmGuid, ModuleManagerGuid,
		CallerGuid, Upcalls, FuncTbl);
}

CSSM_RETURN StaticPlugin::detach(CSSM_MODULE_HANDLE ModuleHandle)
{
	return entries.detach(ModuleHandle);
}


}	// end namespace Security
