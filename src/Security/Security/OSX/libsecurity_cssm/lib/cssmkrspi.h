/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 26, 2025.
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
#ifndef _CSSMKRSPI_H_
#define _CSSMKRSPI_H_  1

#include <Security/cssmtype.h>
#include <Security/cssmkrapi.h> /* CSSM_KRSP_HANDLE */

#ifdef __cplusplus
extern "C" {
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

/* Data types for Key Recovery SPI */

typedef struct DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER cssm_spi_kr_funcs {
    CSSM_RETURN (CSSMKRI *RegistrationRequest)
        (CSSM_KRSP_HANDLE KRSPHandle,
         CSSM_CC_HANDLE KRRegistrationContextHandle,
         const CSSM_CONTEXT *KRRegistrationContext,
         const CSSM_DATA *KRInData,
         const CSSM_ACCESS_CREDENTIALS *AccessCredentials,
         CSSM_KR_POLICY_FLAGS KRFlags,
         sint32 *EstimatedTime,
         CSSM_HANDLE_PTR ReferenceHandle);
    CSSM_RETURN (CSSMKRI *RegistrationRetrieve)
        (CSSM_KRSP_HANDLE KRSPHandle,
         CSSM_HANDLE ReferenceHandle,
         sint32 *EstimatedTime,
         CSSM_KR_PROFILE_PTR KRProfile);
    CSSM_RETURN (CSSMKRI *GenerateRecoveryFields)
        (CSSM_KRSP_HANDLE KRSPHandle,
         CSSM_CC_HANDLE KREnablementContextHandle,
         const CSSM_CONTEXT *KREnablementContext,
         CSSM_CC_HANDLE CryptoContextHandle,
         const CSSM_CONTEXT *CryptoContext,
         const CSSM_DATA *KRSPOptions,
         CSSM_KR_POLICY_FLAGS KRFlags,
         CSSM_DATA_PTR KRFields);
    CSSM_RETURN (CSSMKRI *ProcessRecoveryFields)
        (CSSM_KRSP_HANDLE KRSPHandle,
         CSSM_CC_HANDLE KREnablementContextHandle,
         const CSSM_CONTEXT *KREnablementContext,
         CSSM_CC_HANDLE CryptoContextHandle,
         const CSSM_CONTEXT *CryptoContext,
         const CSSM_DATA *KRSPOptions,
         CSSM_KR_POLICY_FLAGS KRFlags,
         const CSSM_DATA *KRFields);
    CSSM_RETURN (CSSMKRI *RecoveryRequest)
        (CSSM_KRSP_HANDLE KRSPHandle,
         CSSM_CC_HANDLE KRRequestContextHandle,
         const CSSM_CONTEXT *KRRequestContext,
         const CSSM_DATA *KRInData,
         const CSSM_ACCESS_CREDENTIALS *AccessCredentials,
         sint32 *EstimatedTime,
         CSSM_HANDLE_PTR ReferenceHandle);
    CSSM_RETURN (CSSMKRI *RecoveryRetrieve)
        (CSSM_KRSP_HANDLE KRSPHandle,
         CSSM_HANDLE ReferenceHandle,
         sint32 *EstimatedTime,
         CSSM_HANDLE_PTR CacheHandle,
         uint32 *NumberOfRecoveredKeys);
    CSSM_RETURN (CSSMKRI *GetRecoveredObject)
        (CSSM_KRSP_HANDLE KRSPHandle,
         CSSM_HANDLE CacheHandle,
         uint32 IndexInResults,
         CSSM_CSP_HANDLE CSPHandle,
         const CSSM_RESOURCE_CONTROL_CONTEXT *CredAndAclEntry,
         uint32 Flags,
         CSSM_KEY_PTR RecoveredKey,
         CSSM_DATA_PTR OtherInfo);
    CSSM_RETURN (CSSMKRI *RecoveryRequestAbort)
        (CSSM_KRSP_HANDLE KRSPHandle,
         CSSM_HANDLE ResultsHandle);
    CSSM_RETURN (CSSMKRI *PassThrough)
        (CSSM_KRSP_HANDLE KRSPHandle,
         CSSM_CC_HANDLE KeyRecoveryContextHandle,
         const CSSM_CONTEXT *KeyRecoveryContext,
         CSSM_CC_HANDLE CryptoContextHandle,
         const CSSM_CONTEXT *CryptoContext,
         uint32 PassThroughId,
         const void *InputParams,
         void **OutputParams);
} CSSM_SPI_KR_FUNCS DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER, *CSSM_SPI_KR_FUNCS_PTR DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

#pragma clang diagnostic pop

#ifdef __cplusplus
}
#endif

#endif /* _CSSMKRSPI_H_ */
