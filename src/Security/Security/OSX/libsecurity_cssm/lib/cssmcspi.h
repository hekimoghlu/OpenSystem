/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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
#ifndef _CSSMCSPI_H_
#define _CSSMCSPI_H_  1

#include <Security/cssmspi.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

typedef struct DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER cssm_spi_csp_funcs {
    CSSM_RETURN (CSSMCSPI *EventNotify)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CONTEXT_EVENT Event,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context);
    CSSM_RETURN (CSSMCSPI *QuerySize)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context,
         CSSM_BOOL Encrypt,
         uint32 QuerySizeCount,
         CSSM_QUERY_SIZE_DATA_PTR DataBlock);
    CSSM_RETURN (CSSMCSPI *SignData)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context,
         const CSSM_DATA *DataBufs,
         uint32 DataBufCount,
         CSSM_ALGORITHMS DigestAlgorithm,
         CSSM_DATA_PTR Signature);
    CSSM_RETURN (CSSMCSPI *SignDataInit)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context);
    CSSM_RETURN (CSSMCSPI *SignDataUpdate)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_DATA *DataBufs,
         uint32 DataBufCount);
    CSSM_RETURN (CSSMCSPI *SignDataFinal)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         CSSM_DATA_PTR Signature);
    CSSM_RETURN (CSSMCSPI *VerifyData)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context,
         const CSSM_DATA *DataBufs,
         uint32 DataBufCount,
         CSSM_ALGORITHMS DigestAlgorithm,
         const CSSM_DATA *Signature);
    CSSM_RETURN (CSSMCSPI *VerifyDataInit)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context);
    CSSM_RETURN (CSSMCSPI *VerifyDataUpdate)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_DATA *DataBufs,
         uint32 DataBufCount);
    CSSM_RETURN (CSSMCSPI *VerifyDataFinal)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_DATA *Signature);
    CSSM_RETURN (CSSMCSPI *DigestData)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context,
         const CSSM_DATA *DataBufs,
         uint32 DataBufCount,
         CSSM_DATA_PTR Digest);
    CSSM_RETURN (CSSMCSPI *DigestDataInit)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context);
    CSSM_RETURN (CSSMCSPI *DigestDataUpdate)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_DATA *DataBufs,
         uint32 DataBufCount);
    CSSM_RETURN (CSSMCSPI *DigestDataClone)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         CSSM_CC_HANDLE ClonedCCHandle);
    CSSM_RETURN (CSSMCSPI *DigestDataFinal)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         CSSM_DATA_PTR Digest);
    CSSM_RETURN (CSSMCSPI *GenerateMac)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context,
         const CSSM_DATA *DataBufs,
         uint32 DataBufCount,
         CSSM_DATA_PTR Mac);
    CSSM_RETURN (CSSMCSPI *GenerateMacInit)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context);
    CSSM_RETURN (CSSMCSPI *GenerateMacUpdate)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_DATA *DataBufs,
         uint32 DataBufCount);
    CSSM_RETURN (CSSMCSPI *GenerateMacFinal)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         CSSM_DATA_PTR Mac);
    CSSM_RETURN (CSSMCSPI *VerifyMac)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context,
         const CSSM_DATA *DataBufs,
         uint32 DataBufCount,
         const CSSM_DATA *Mac);
    CSSM_RETURN (CSSMCSPI *VerifyMacInit)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context);
    CSSM_RETURN (CSSMCSPI *VerifyMacUpdate)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_DATA *DataBufs,
         uint32 DataBufCount);
    CSSM_RETURN (CSSMCSPI *VerifyMacFinal)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_DATA *Mac);
    CSSM_RETURN (CSSMCSPI *EncryptData)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context,
         const CSSM_DATA *ClearBufs,
         uint32 ClearBufCount,
         CSSM_DATA_PTR CipherBufs,
         uint32 CipherBufCount,
         CSSM_SIZE *bytesEncrypted,
         CSSM_DATA_PTR RemData,
         CSSM_PRIVILEGE Privilege);
    CSSM_RETURN (CSSMCSPI *EncryptDataInit)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context,
         CSSM_PRIVILEGE Privilege);
    CSSM_RETURN (CSSMCSPI *EncryptDataUpdate)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_DATA *ClearBufs,
         uint32 ClearBufCount,
         CSSM_DATA_PTR CipherBufs,
         uint32 CipherBufCount,
         CSSM_SIZE *bytesEncrypted);
    CSSM_RETURN (CSSMCSPI *EncryptDataFinal)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         CSSM_DATA_PTR RemData);
    CSSM_RETURN (CSSMCSPI *DecryptData)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context,
         const CSSM_DATA *CipherBufs,
         uint32 CipherBufCount,
         CSSM_DATA_PTR ClearBufs,
         uint32 ClearBufCount,
         CSSM_SIZE *bytesDecrypted,
         CSSM_DATA_PTR RemData,
         CSSM_PRIVILEGE Privilege);
    CSSM_RETURN (CSSMCSPI *DecryptDataInit)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context,
         CSSM_PRIVILEGE Privilege);
    CSSM_RETURN (CSSMCSPI *DecryptDataUpdate)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_DATA *CipherBufs,
         uint32 CipherBufCount,
         CSSM_DATA_PTR ClearBufs,
         uint32 ClearBufCount,
         CSSM_SIZE *bytesDecrypted);
    CSSM_RETURN (CSSMCSPI *DecryptDataFinal)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         CSSM_DATA_PTR RemData);
    CSSM_RETURN (CSSMCSPI *QueryKeySizeInBits)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context,
         const CSSM_KEY *Key,
         CSSM_KEY_SIZE_PTR KeySize);
    CSSM_RETURN (CSSMCSPI *GenerateKey)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context,
         uint32 KeyUsage,
         uint32 KeyAttr,
         const CSSM_DATA *KeyLabel,
         const CSSM_RESOURCE_CONTROL_CONTEXT *CredAndAclEntry,
         CSSM_KEY_PTR Key,
         CSSM_PRIVILEGE Privilege);
    CSSM_RETURN (CSSMCSPI *GenerateKeyPair)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context,
         uint32 PublicKeyUsage,
         uint32 PublicKeyAttr,
         const CSSM_DATA *PublicKeyLabel,
         CSSM_KEY_PTR PublicKey,
         uint32 PrivateKeyUsage,
         uint32 PrivateKeyAttr,
         const CSSM_DATA *PrivateKeyLabel,
         const CSSM_RESOURCE_CONTROL_CONTEXT *CredAndAclEntry,
         CSSM_KEY_PTR PrivateKey,
         CSSM_PRIVILEGE Privilege);
   CSSM_RETURN (CSSMCSPI *GenerateRandom)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context,
         CSSM_DATA_PTR RandomNumber);
    CSSM_RETURN (CSSMCSPI *GenerateAlgorithmParams)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context,
         uint32 ParamBits,
         CSSM_DATA_PTR Param,
         uint32 *NumberOfUpdatedAttibutes,
         CSSM_CONTEXT_ATTRIBUTE_PTR *UpdatedAttributes);
    CSSM_RETURN (CSSMCSPI *WrapKey)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context,
         const CSSM_ACCESS_CREDENTIALS *AccessCred,
         const CSSM_KEY *Key,
         const CSSM_DATA *DescriptiveData,
         CSSM_WRAP_KEY_PTR WrappedKey,
         CSSM_PRIVILEGE Privilege);
    CSSM_RETURN (CSSMCSPI *UnwrapKey)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context,
         const CSSM_KEY *PublicKey,
         const CSSM_WRAP_KEY *WrappedKey,
         uint32 KeyUsage,
         uint32 KeyAttr,
         const CSSM_DATA *KeyLabel,
         const CSSM_RESOURCE_CONTROL_CONTEXT *CredAndAclEntry,
         CSSM_KEY_PTR UnwrappedKey,
         CSSM_DATA_PTR DescriptiveData,
         CSSM_PRIVILEGE Privilege);
    CSSM_RETURN (CSSMCSPI *DeriveKey)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context,
         CSSM_DATA_PTR Param,
         uint32 KeyUsage,
         uint32 KeyAttr,
         const CSSM_DATA *KeyLabel,
         const CSSM_RESOURCE_CONTROL_CONTEXT *CredAndAclEntry,
         CSSM_KEY_PTR DerivedKey);
    CSSM_RETURN (CSSMCSPI *FreeKey)
        (CSSM_CSP_HANDLE CSPHandle,
         const CSSM_ACCESS_CREDENTIALS *AccessCred,
         CSSM_KEY_PTR KeyPtr,
         CSSM_BOOL Delete);
    CSSM_RETURN (CSSMCSPI *PassThrough)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_CONTEXT *Context,
         uint32 PassThroughId,
         const void *InData,
         void **OutData);
    CSSM_RETURN (CSSMCSPI *Login)
        (CSSM_CSP_HANDLE CSPHandle,
         const CSSM_ACCESS_CREDENTIALS *AccessCred,
         const CSSM_DATA *LoginName,
         const void *Reserved);
    CSSM_RETURN (CSSMCSPI *Logout)
        (CSSM_CSP_HANDLE CSPHandle);
    CSSM_RETURN (CSSMCSPI *ChangeLoginAcl)
        (CSSM_CSP_HANDLE CSPHandle,
         const CSSM_ACCESS_CREDENTIALS *AccessCred,
         const CSSM_ACL_EDIT *AclEdit);
    CSSM_RETURN (CSSMCSPI *ObtainPrivateKeyFromPublicKey)
        (CSSM_CSP_HANDLE CSPHandle,
         const CSSM_KEY *PublicKey,
         CSSM_KEY_PTR PrivateKey);
    CSSM_RETURN (CSSMCSPI *RetrieveUniqueId)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_DATA_PTR UniqueID);
    CSSM_RETURN (CSSMCSPI *RetrieveCounter)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_DATA_PTR Counter);
    CSSM_RETURN (CSSMCSPI *VerifyDevice)
        (CSSM_CSP_HANDLE CSPHandle,
         const CSSM_DATA *DeviceCert);
    CSSM_RETURN (CSSMCSPI *GetTimeValue)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_ALGORITHMS TimeAlgorithm,
         CSSM_DATA *TimeData);
    CSSM_RETURN (CSSMCSPI *GetOperationalStatistics)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_CSP_OPERATIONAL_STATISTICS *Statistics);
    CSSM_RETURN (CSSMCSPI *GetLoginAcl)
        (CSSM_CSP_HANDLE CSPHandle,
         const CSSM_STRING *SelectionTag,
         uint32 *NumberOfAclInfos,
         CSSM_ACL_ENTRY_INFO_PTR *AclInfos);
    CSSM_RETURN (CSSMCSPI *GetKeyAcl)
        (CSSM_CSP_HANDLE CSPHandle,
         const CSSM_KEY *Key,
         const CSSM_STRING *SelectionTag,
         uint32 *NumberOfAclInfos,
         CSSM_ACL_ENTRY_INFO_PTR *AclInfos);
    CSSM_RETURN (CSSMCSPI *ChangeKeyAcl)
        (CSSM_CSP_HANDLE CSPHandle,
         const CSSM_ACCESS_CREDENTIALS *AccessCred,
         const CSSM_ACL_EDIT *AclEdit,
         const CSSM_KEY *Key);
    CSSM_RETURN (CSSMCSPI *GetKeyOwner)
        (CSSM_CSP_HANDLE CSPHandle,
         const CSSM_KEY *Key,
         CSSM_ACL_OWNER_PROTOTYPE_PTR Owner);
    CSSM_RETURN (CSSMCSPI *ChangeKeyOwner)
        (CSSM_CSP_HANDLE CSPHandle,
         const CSSM_ACCESS_CREDENTIALS *AccessCred,
         const CSSM_KEY *Key,
         const CSSM_ACL_OWNER_PROTOTYPE *NewOwner);
    CSSM_RETURN (CSSMCSPI *GetLoginOwner)
        (CSSM_CSP_HANDLE CSPHandle,
         CSSM_ACL_OWNER_PROTOTYPE_PTR Owner);
    CSSM_RETURN (CSSMCSPI *ChangeLoginOwner)
        (CSSM_CSP_HANDLE CSPHandle,
         const CSSM_ACCESS_CREDENTIALS *AccessCred,
         const CSSM_ACL_OWNER_PROTOTYPE *NewOwner);
} CSSM_SPI_CSP_FUNCS DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER, *CSSM_SPI_CSP_FUNCS_PTR DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

#pragma clang diagnostic pop

#ifdef __cplusplus
}
#endif

#endif /* _CSSMCSPI_H_ */
