/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 8, 2024.
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
// cssmint - cssm combined internal headers
//
#ifndef _H_CSSMINT
#define _H_CSSMINT

#include <Security/cssm.h>
#include <Security/cssmspi.h>
#include <security_utilities/utilities.h>
#include <security_utilities/alloc.h>
#include <security_utilities/threading.h>


//
// Forward types
//
class CssmManager;
class Module;
class Attachment;


//
// Typedefs for fixed SPI entries
//
typedef CSSM_RETURN CSSMSPI
CSSM_SPI_ModuleLoadFunction (const CSSM_GUID *CssmGuid,
                             const CSSM_GUID *ModuleGuid,
                             CSSM_SPI_ModuleEventHandler CssmNotifyCallback,
                             void *CssmNotifyCallbackCtx);

typedef CSSM_RETURN CSSMSPI
CSSM_SPI_ModuleUnloadFunction (const CSSM_GUID *CssmGuid,
                               const CSSM_GUID *ModuleGuid,
                               CSSM_SPI_ModuleEventHandler CssmNotifyCallback,
                               void *CssmNotifyCallbackCtx);

typedef CSSM_RETURN CSSMSPI
CSSM_SPI_ModuleAttachFunction (const CSSM_GUID *ModuleGuid,
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

typedef CSSM_RETURN CSSMSPI
CSSM_SPI_ModuleDetachFunction (CSSM_MODULE_HANDLE ModuleHandle);



#endif //_H_CSSMINT
