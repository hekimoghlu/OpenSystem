/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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
#ifndef _CSSMSPI_H_
#define _CSSMSPI_H_  1

#include <Security/cssmtype.h>
#include <Security/cssmspi.h> /* CSSM_UPCALLS_PTR */

#ifdef __cplusplus
extern "C" {
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

typedef CSSM_RETURN (CSSMAPI *CSSM_SPI_ModuleEventHandler)
    (const CSSM_GUID *ModuleGuid,
     void *CssmNotifyCallbackCtx,
     uint32 SubserviceId,
     CSSM_SERVICE_TYPE ServiceType,
     CSSM_MODULE_EVENT EventType) DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

typedef uint32 CSSM_CONTEXT_EVENT;
enum {
    CSSM_CONTEXT_EVENT_CREATE = 1,
    CSSM_CONTEXT_EVENT_DELETE = 2,
    CSSM_CONTEXT_EVENT_UPDATE = 3
};

typedef struct DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER cssm_module_funcs {
    CSSM_SERVICE_TYPE ServiceType;
    uint32 NumberOfServiceFuncs;
    const CSSM_PROC_ADDR *ServiceFuncs;
} CSSM_MODULE_FUNCS DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER, *CSSM_MODULE_FUNCS_PTR DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

typedef void *(CSSMAPI *CSSM_UPCALLS_MALLOC)
    (CSSM_HANDLE AddInHandle,
     size_t size) DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

typedef void (CSSMAPI *CSSM_UPCALLS_FREE)
    (CSSM_HANDLE AddInHandle,
     void *memblock) DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

typedef void *(CSSMAPI *CSSM_UPCALLS_REALLOC)
    (CSSM_HANDLE AddInHandle,
     void *memblock,
     size_t size) DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

typedef void *(CSSMAPI *CSSM_UPCALLS_CALLOC)
    (CSSM_HANDLE AddInHandle,
     size_t num,
     size_t size) DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

typedef struct DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER cssm_upcalls {
    CSSM_UPCALLS_MALLOC malloc_func;
    CSSM_UPCALLS_FREE free_func;
    CSSM_UPCALLS_REALLOC realloc_func;
    CSSM_UPCALLS_CALLOC calloc_func;
    CSSM_RETURN (CSSMAPI *CcToHandle_func)
        (CSSM_CC_HANDLE Cc,
         CSSM_MODULE_HANDLE_PTR ModuleHandle);
    CSSM_RETURN (CSSMAPI *GetModuleInfo_func)
        (CSSM_MODULE_HANDLE Module,
         CSSM_GUID_PTR Guid,
         CSSM_VERSION_PTR Version,
         uint32 *SubServiceId,
         CSSM_SERVICE_TYPE *SubServiceType,
         CSSM_ATTACH_FLAGS *AttachFlags,
         CSSM_KEY_HIERARCHY *KeyHierarchy,
         CSSM_API_MEMORY_FUNCS_PTR AttachedMemFuncs,
         CSSM_FUNC_NAME_ADDR_PTR FunctionTable,
         uint32 NumFunctions);
} CSSM_UPCALLS DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER, *CSSM_UPCALLS_PTR DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

#pragma clang diagnostic pop

#ifdef __cplusplus
}
#endif

#endif /* _CSSMSPI_H_ */
