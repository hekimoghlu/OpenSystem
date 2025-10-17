/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 5, 2023.
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
// pluginspi - "roof" level entry points into a CSSM plugin.
//
// This file is meant to be included into the top-level source file
// for a CSSM plugin written to the C++ alternate interface.
// It contains actual code that defines the four required entry points.
//
#include <security_cdsa_utilities/cssmbridge.h>


//
// Provide some flexibility for the includer
//
#if !defined(SPIPREFIX)
# define SPIPREFIX	extern "C" CSSMSPI
#endif

#if !defined(SPINAME)
# define SPINAME(s) s
#endif

SPIPREFIX CSSM_RETURN SPINAME(CSSM_SPI_ModuleLoad) (const CSSM_GUID *CssmGuid,
                                                    const CSSM_GUID *ModuleGuid,
                                                    CSSM_SPI_ModuleEventHandler CssmNotifyCallback,
                                                    void *CssmNotifyCallbackCtx);

SPIPREFIX CSSM_RETURN SPINAME(CSSM_SPI_ModuleLoad) (const CSSM_GUID *CssmGuid,
    const CSSM_GUID *ModuleGuid,
    CSSM_SPI_ModuleEventHandler CssmNotifyCallback,
    void *CssmNotifyCallbackCtx)
{
    BEGIN_API_NO_METRICS
    plugin().moduleLoad(Guid::required(CssmGuid),
        Guid::required(ModuleGuid),
        ModuleCallback(CssmNotifyCallback, CssmNotifyCallbackCtx));
    END_API_NO_METRICS(CSSM)
}

SPIPREFIX CSSM_RETURN SPINAME(CSSM_SPI_ModuleUnload) (const CSSM_GUID *CssmGuid,
                                                      const CSSM_GUID *ModuleGuid,
                                                      CSSM_SPI_ModuleEventHandler CssmNotifyCallback,
                                                      void *CssmNotifyCallbackCtx);

SPIPREFIX CSSM_RETURN SPINAME(CSSM_SPI_ModuleUnload) (const CSSM_GUID *CssmGuid,
    const CSSM_GUID *ModuleGuid,
    CSSM_SPI_ModuleEventHandler CssmNotifyCallback,
    void *CssmNotifyCallbackCtx)
{
    BEGIN_API_NO_METRICS
    plugin().moduleUnload(Guid::required(CssmGuid),
        Guid::required(ModuleGuid),
        ModuleCallback(CssmNotifyCallback, CssmNotifyCallbackCtx));
    END_API_NO_METRICS(CSSM)
}

SPIPREFIX CSSM_RETURN SPINAME(CSSM_SPI_ModuleAttach) (const CSSM_GUID *ModuleGuid,
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
                                                      CSSM_MODULE_FUNCS_PTR *FuncTbl);

SPIPREFIX CSSM_RETURN SPINAME(CSSM_SPI_ModuleAttach) (const CSSM_GUID *ModuleGuid,
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
    BEGIN_API_NO_METRICS
    plugin().moduleAttach(ModuleHandle,
        Guid::required(CssmGuid),
        Guid::required(ModuleGuid),
        Guid::required(ModuleManagerGuid),
        Guid::required(CallerGuid),
        *Version,
        SubserviceID,
        SubServiceType,
        AttachFlags,
        KeyHierarchy,
        Required(Upcalls),
        Required(FuncTbl));
    END_API_NO_METRICS(CSSM)
}

SPIPREFIX CSSM_RETURN SPINAME(CSSM_SPI_ModuleDetach) (CSSM_MODULE_HANDLE ModuleHandle);

SPIPREFIX CSSM_RETURN SPINAME(CSSM_SPI_ModuleDetach) (CSSM_MODULE_HANDLE ModuleHandle)
{
    BEGIN_API_NO_METRICS
    plugin().moduleDetach(ModuleHandle);
    END_API_NO_METRICS(CSSM)
}
