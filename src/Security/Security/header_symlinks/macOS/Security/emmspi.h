/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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
#ifndef _EMMSPI_H_
#define _EMMSPI_H_  1

#include <Security/cssmtype.h>
#include <Security/emmtype.h> /* CSSM_MANAGER_EVENT_NOTIFICATION */
#include <Security/cssmspi.h> /* CSSM_UPCALLS_PTR */

#ifdef __cplusplus
extern "C" {
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

typedef struct DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER cssm_state_funcs {
    CSSM_RETURN (CSSMAPI *cssm_GetAttachFunctions)
        (CSSM_MODULE_HANDLE hAddIn,
         CSSM_SERVICE_MASK AddinType,
         void **SPFunctions,
         CSSM_GUID_PTR Guid,
	 CSSM_BOOL *Serialized);
    CSSM_RETURN (CSSMAPI *cssm_ReleaseAttachFunctions)
        (CSSM_MODULE_HANDLE hAddIn);
    CSSM_RETURN (CSSMAPI *cssm_GetAppMemoryFunctions)
        (CSSM_MODULE_HANDLE hAddIn,
         CSSM_UPCALLS_PTR UpcallTable);
    CSSM_RETURN (CSSMAPI *cssm_IsFuncCallValid)
        (CSSM_MODULE_HANDLE hAddin,
         CSSM_PROC_ADDR SrcAddress,
         CSSM_PROC_ADDR DestAddress,
         CSSM_PRIVILEGE InPriv,
         CSSM_PRIVILEGE *OutPriv,
         CSSM_BITMASK Hints,
         CSSM_BOOL *IsOK);
    CSSM_RETURN (CSSMAPI *cssm_DeregisterManagerServices)
        (const CSSM_GUID *GUID);
    CSSM_RETURN (CSSMAPI *cssm_DeliverModuleManagerEvent)
        (const CSSM_MANAGER_EVENT_NOTIFICATION *EventDescription);
} CSSM_STATE_FUNCS DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER, *CSSM_STATE_FUNCS_PTR DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

typedef struct DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER cssm_manager_registration_info {
    /* loading, unloading, dispatch table, and event notification */
    CSSM_RETURN (CSSMAPI *Initialize)
        (uint32 VerMajor,
         uint32 VerMinor);
    CSSM_RETURN (CSSMAPI *Terminate) (void);
    CSSM_RETURN (CSSMAPI *RegisterDispatchTable)
        (CSSM_STATE_FUNCS_PTR CssmStateCallTable);
    CSSM_RETURN (CSSMAPI *DeregisterDispatchTable) (void);
    CSSM_RETURN (CSSMAPI *EventNotifyManager)
        (const CSSM_MANAGER_EVENT_NOTIFICATION *EventDescription);
    CSSM_RETURN (CSSMAPI *RefreshFunctionTable)
        (CSSM_FUNC_NAME_ADDR_PTR FuncNameAddrPtr,
         uint32 NumOfFuncNameAddr);
} CSSM_MANAGER_REGISTRATION_INFO DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER, *CSSM_MANAGER_REGISTRATION_INFO_PTR DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

enum {
	CSSM_HINT_NONE =			0,
	CSSM_HINT_ADDRESS_APP = 	1 << 0,
	CSSM_HINT_ADDRESS_SP =		1 << 1
};

#pragma clang diagnostic pop

#ifdef __cplusplus
}
#endif

#endif /* _EMMSPI_H_ */
