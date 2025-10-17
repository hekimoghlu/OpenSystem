/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 20, 2024.
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
#ifndef _CSSMACI_H_
#define _CSSMACI_H_  1

#include <Security/cssmtype.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

typedef struct DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER cssm_spi_ac_funcs {
    CSSM_RETURN (CSSMACI *AuthCompute)
        (CSSM_AC_HANDLE ACHandle,
         const CSSM_TUPLEGROUP *BaseAuthorizations,
         const CSSM_TUPLEGROUP *Credentials,
         uint32 NumberOfRequestors,
         const CSSM_LIST *Requestors,
         const CSSM_LIST *RequestedAuthorizationPeriod,
         const CSSM_LIST *RequestedAuthorization,
         CSSM_TUPLEGROUP_PTR AuthorizationResult);
    CSSM_RETURN (CSSMACI *PassThrough)
        (CSSM_AC_HANDLE ACHandle,
         CSSM_TP_HANDLE TPHandle,
         CSSM_CL_HANDLE CLHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_DL_DB_LIST *DBList,
         uint32 PassThroughId,
         const void *InputParams,
         void **OutputParams);
} CSSM_SPI_AC_FUNCS DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER, *CSSM_SPI_AC_FUNCS_PTR DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

#pragma clang diagnostic pop

#ifdef __cplusplus
}
#endif

#endif /* _CSSMACI_H_ */
