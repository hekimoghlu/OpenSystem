/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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
#ifndef _CSSMDLI_H_
#define _CSSMDLI_H_  1

#include <Security/cssmtype.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

typedef struct DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER cssm_spi_dl_funcs {
    CSSM_RETURN (CSSMDLI *DbOpen)
        (CSSM_DL_HANDLE DLHandle,
         const char *DbName,
         const CSSM_NET_ADDRESS *DbLocation,
         CSSM_DB_ACCESS_TYPE AccessRequest,
         const CSSM_ACCESS_CREDENTIALS *AccessCred,
         const void *OpenParameters,
         CSSM_DB_HANDLE *DbHandle);
    CSSM_RETURN (CSSMDLI *DbClose)
        (CSSM_DL_DB_HANDLE DLDBHandle);
    CSSM_RETURN (CSSMDLI *DbCreate)
        (CSSM_DL_HANDLE DLHandle,
         const char *DbName,
         const CSSM_NET_ADDRESS *DbLocation,
         const CSSM_DBINFO *DBInfo,
         CSSM_DB_ACCESS_TYPE AccessRequest,
         const CSSM_RESOURCE_CONTROL_CONTEXT *CredAndAclEntry,
         const void *OpenParameters,
         CSSM_DB_HANDLE *DbHandle);
    CSSM_RETURN (CSSMDLI *DbDelete)
        (CSSM_DL_HANDLE DLHandle,
         const char *DbName,
         const CSSM_NET_ADDRESS *DbLocation,
         const CSSM_ACCESS_CREDENTIALS *AccessCred);
    CSSM_RETURN (CSSMDLI *CreateRelation)
        (CSSM_DL_DB_HANDLE DLDBHandle,
         CSSM_DB_RECORDTYPE RelationID,
         const char *RelationName,
         uint32 NumberOfAttributes,
         const CSSM_DB_SCHEMA_ATTRIBUTE_INFO *pAttributeInfo,
         uint32 NumberOfIndexes,
         const CSSM_DB_SCHEMA_INDEX_INFO *pIndexInfo);
    CSSM_RETURN (CSSMDLI *DestroyRelation)
        (CSSM_DL_DB_HANDLE DLDBHandle,
         CSSM_DB_RECORDTYPE RelationID);
    CSSM_RETURN (CSSMDLI *Authenticate)
        (CSSM_DL_DB_HANDLE DLDBHandle,
         CSSM_DB_ACCESS_TYPE AccessRequest,
         const CSSM_ACCESS_CREDENTIALS *AccessCred);
    CSSM_RETURN (CSSMDLI *GetDbAcl)
        (CSSM_DL_DB_HANDLE DLDBHandle,
         const CSSM_STRING *SelectionTag,
         uint32 *NumberOfAclInfos,
         CSSM_ACL_ENTRY_INFO_PTR *AclInfos);
    CSSM_RETURN (CSSMDLI *ChangeDbAcl)
        (CSSM_DL_DB_HANDLE DLDBHandle,
         const CSSM_ACCESS_CREDENTIALS *AccessCred,
         const CSSM_ACL_EDIT *AclEdit);
    CSSM_RETURN (CSSMDLI *GetDbOwner)
        (CSSM_DL_DB_HANDLE DLDBHandle,
         CSSM_ACL_OWNER_PROTOTYPE_PTR Owner);
    CSSM_RETURN (CSSMDLI *ChangeDbOwner)
        (CSSM_DL_DB_HANDLE DLDBHandle,
         const CSSM_ACCESS_CREDENTIALS *AccessCred,
         const CSSM_ACL_OWNER_PROTOTYPE *NewOwner);
    CSSM_RETURN (CSSMDLI *GetDbNames)
        (CSSM_DL_HANDLE DLHandle,
         CSSM_NAME_LIST_PTR *NameList);
    CSSM_RETURN (CSSMDLI *GetDbNameFromHandle)
        (CSSM_DL_DB_HANDLE DLDBHandle,
         char **DbName);
    CSSM_RETURN (CSSMDLI *FreeNameList)
        (CSSM_DL_HANDLE DLHandle,
         CSSM_NAME_LIST_PTR NameList);
    CSSM_RETURN (CSSMDLI *DataInsert)
        (CSSM_DL_DB_HANDLE DLDBHandle,
         CSSM_DB_RECORDTYPE RecordType,
         const CSSM_DB_RECORD_ATTRIBUTE_DATA *Attributes,
         const CSSM_DATA *Data,
         CSSM_DB_UNIQUE_RECORD_PTR *UniqueId);
    CSSM_RETURN (CSSMDLI *DataDelete)
        (CSSM_DL_DB_HANDLE DLDBHandle,
         const CSSM_DB_UNIQUE_RECORD *UniqueRecordIdentifier);
    CSSM_RETURN (CSSMDLI *DataModify)
        (CSSM_DL_DB_HANDLE DLDBHandle,
         CSSM_DB_RECORDTYPE RecordType,
         CSSM_DB_UNIQUE_RECORD_PTR UniqueRecordIdentifier,
         const CSSM_DB_RECORD_ATTRIBUTE_DATA *AttributesToBeModified,
         const CSSM_DATA *DataToBeModified,
         CSSM_DB_MODIFY_MODE ModifyMode);
    CSSM_RETURN (CSSMDLI *DataGetFirst)
        (CSSM_DL_DB_HANDLE DLDBHandle,
         const CSSM_QUERY *Query,
         CSSM_HANDLE_PTR ResultsHandle,
         CSSM_DB_RECORD_ATTRIBUTE_DATA_PTR Attributes,
         CSSM_DATA_PTR Data,
         CSSM_DB_UNIQUE_RECORD_PTR *UniqueId);
    CSSM_RETURN (CSSMDLI *DataGetNext)
        (CSSM_DL_DB_HANDLE DLDBHandle,
         CSSM_HANDLE ResultsHandle,
         CSSM_DB_RECORD_ATTRIBUTE_DATA_PTR Attributes,
         CSSM_DATA_PTR Data,
         CSSM_DB_UNIQUE_RECORD_PTR *UniqueId);
    CSSM_RETURN (CSSMDLI *DataAbortQuery)
        (CSSM_DL_DB_HANDLE DLDBHandle,
         CSSM_HANDLE ResultsHandle);
    CSSM_RETURN (CSSMDLI *DataGetFromUniqueRecordId)
        (CSSM_DL_DB_HANDLE DLDBHandle,
         const CSSM_DB_UNIQUE_RECORD *UniqueRecord,
         CSSM_DB_RECORD_ATTRIBUTE_DATA_PTR Attributes,
         CSSM_DATA_PTR Data);
    CSSM_RETURN (CSSMDLI *FreeUniqueRecord)
        (CSSM_DL_DB_HANDLE DLDBHandle,
         CSSM_DB_UNIQUE_RECORD_PTR UniqueRecord);
    CSSM_RETURN (CSSMDLI *PassThrough)
        (CSSM_DL_DB_HANDLE DLDBHandle,
         uint32 PassThroughId,
         const void *InputParams,
         void **OutputParams);
} CSSM_SPI_DL_FUNCS DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER, *CSSM_SPI_DL_FUNCS_PTR DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

#pragma clang diagnostic pop

#ifdef __cplusplus
}
#endif

#endif /* _CSSMDLI_H_ */
