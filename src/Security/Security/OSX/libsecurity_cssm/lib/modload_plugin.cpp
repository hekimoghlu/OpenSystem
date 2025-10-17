/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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
// modload_plugin - loader interface for dynamically loaded plugin modules
//
#include "modload_plugin.h"
#include <security_cdsa_utilities/cssmerrors.h>


namespace Security {


//
// During construction, a LoadablePlugin loads itself into memory and locates
// the canonical (CDSA defined) four entrypoints. If anything fails, we throw.
//    
LoadablePlugin::LoadablePlugin(const char *path) : LoadableBundle(path)
{
	secinfo("cssm", "LoadablePlugin(%s)", path);
    if (!allowableModulePath(path)) {
        secinfo("cssm", "LoadablePlugin(): not loaded; plugin in non-standard location: %s", path);
        CssmError::throwMe(CSSMERR_CSSM_ADDIN_AUTHENTICATE_FAILED);
    }
    load();
}


//
// Loading and unloading devolves directly onto LoadableBundle
//
void LoadablePlugin::load()
{
	secinfo("cssm", "LoadablePlugin::load() path %s", path().c_str());
    LoadableBundle::load();
    findFunction(mFunctions.load, "CSSM_SPI_ModuleLoad");
    findFunction(mFunctions.attach, "CSSM_SPI_ModuleAttach");
    findFunction(mFunctions.detach, "CSSM_SPI_ModuleDetach");
    findFunction(mFunctions.unload, "CSSM_SPI_ModuleUnload");
}

void LoadablePlugin::unload()
{
	secinfo("cssm", "LoadablePlugin::unload() path %s", path().c_str());
	/* skipping for workaround for radar 3774226 
    LoadableBundle::unload(); */ 
}

bool LoadablePlugin::isLoaded() const
{
    return LoadableBundle::isLoaded();
}


//
// Pass module entry points to the statically linked functions
//
CSSM_RETURN LoadablePlugin::load(const CSSM_GUID *CssmGuid,
                             const CSSM_GUID *ModuleGuid,
                             CSSM_SPI_ModuleEventHandler CssmNotifyCallback,
                             void *CssmNotifyCallbackCtx)
{
	secinfo("cssm", "LoadablePlugin::load(guid,...) path %s", path().c_str());
	return mFunctions.load(CssmGuid, ModuleGuid,
		CssmNotifyCallback, CssmNotifyCallbackCtx);
}

CSSM_RETURN LoadablePlugin::unload(const CSSM_GUID *CssmGuid,
                             const CSSM_GUID *ModuleGuid,
                             CSSM_SPI_ModuleEventHandler CssmNotifyCallback,
                             void *CssmNotifyCallbackCtx)
{
	secinfo("cssm", "LoadablePlugin::unload(guid,...) path %s", path().c_str());
	return mFunctions.unload(CssmGuid, ModuleGuid,
		CssmNotifyCallback, CssmNotifyCallbackCtx);
}

CSSM_RETURN LoadablePlugin::attach(const CSSM_GUID *ModuleGuid,
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
	return mFunctions.attach(ModuleGuid, Version, SubserviceID, SubServiceType,
		AttachFlags, ModuleHandle, KeyHierarchy, CssmGuid, ModuleManagerGuid,
		CallerGuid, Upcalls, FuncTbl);
}

CSSM_RETURN LoadablePlugin::detach(CSSM_MODULE_HANDLE ModuleHandle)
{
	return mFunctions.detach(ModuleHandle);
}

bool LoadablePlugin::allowableModulePath(const char *path) {
    // True if module path is in default location
    const char *loadablePrefix="/System/Library/Security/";
    return (strncmp(loadablePrefix,path,strlen(loadablePrefix)) == 0);
}

}	// end namespace Security
