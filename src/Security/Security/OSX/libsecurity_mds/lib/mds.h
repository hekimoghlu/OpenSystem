/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 28, 2025.
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
/*
   File:      mds.h

   Contains:  Module Directory Services Data Types and API.

   Copyright (c) 1999-2000,2011,2014 Apple Inc. All Rights Reserved.
*/

#ifndef _MDS_H_
#define _MDS_H_  1

#include <Security/cssmtype.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

typedef CSSM_DL_HANDLE MDS_HANDLE;

typedef CSSM_DL_DB_HANDLE MDS_DB_HANDLE DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

typedef struct DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER mds_funcs {
    CSSM_RETURN (CSSMAPI *DbOpen)
        (MDS_HANDLE MdsHandle,
         const char *DbName,
         const CSSM_NET_ADDRESS *DbLocation,
         CSSM_DB_ACCESS_TYPE AccessRequest,
         const CSSM_ACCESS_CREDENTIALS *AccessCred,
         const void *OpenParameters,
         CSSM_DB_HANDLE *hMds);

    CSSM_RETURN (CSSMAPI *DbClose)
        (MDS_DB_HANDLE MdsDbHandle);

    CSSM_RETURN (CSSMAPI *GetDbNames)
        (MDS_HANDLE MdsHandle,
         CSSM_NAME_LIST_PTR *NameList);

    CSSM_RETURN (CSSMAPI *GetDbNameFromHandle)
        (MDS_DB_HANDLE MdsDbHandle,
         char **DbName);

    CSSM_RETURN (CSSMAPI *FreeNameList)
        (MDS_HANDLE MdsHandle,
         CSSM_NAME_LIST_PTR NameList);

    CSSM_RETURN (CSSMAPI *DataInsert)
        (MDS_DB_HANDLE MdsDbHandle,
         CSSM_DB_RECORDTYPE RecordType,
         const CSSM_DB_RECORD_ATTRIBUTE_DATA *Attributes,
         const CSSM_DATA *Data,
         CSSM_DB_UNIQUE_RECORD_PTR *UniqueId);

    CSSM_RETURN (CSSMAPI *DataDelete)
        (MDS_DB_HANDLE MdsDbHandle,
         const CSSM_DB_UNIQUE_RECORD *UniqueRecordIdentifier);

    CSSM_RETURN (CSSMAPI *DataModify)
        (MDS_DB_HANDLE MdsDbHandle,
         CSSM_DB_RECORDTYPE RecordType,
         CSSM_DB_UNIQUE_RECORD_PTR UniqueRecordIdentifier,
         const CSSM_DB_RECORD_ATTRIBUTE_DATA *AttributesToBeModified,
         const CSSM_DATA *DataToBeModified,
         CSSM_DB_MODIFY_MODE ModifyMode);

    CSSM_RETURN (CSSMAPI *DataGetFirst)
        (MDS_DB_HANDLE MdsDbHandle,
         const CSSM_QUERY *Query,
         CSSM_HANDLE_PTR ResultsHandle,
         CSSM_DB_RECORD_ATTRIBUTE_DATA_PTR Attributes,
         CSSM_DATA_PTR Data,
         CSSM_DB_UNIQUE_RECORD_PTR *UniqueId);

    CSSM_RETURN (CSSMAPI *DataGetNext)
        (MDS_DB_HANDLE MdsDbHandle,
         CSSM_HANDLE ResultsHandle,
         CSSM_DB_RECORD_ATTRIBUTE_DATA_PTR Attributes,
         CSSM_DATA_PTR Data,
         CSSM_DB_UNIQUE_RECORD_PTR *UniqueId);

    CSSM_RETURN (CSSMAPI *DataAbortQuery)
        (MDS_DB_HANDLE MdsDbHandle,
         CSSM_HANDLE ResultsHandle);

    CSSM_RETURN (CSSMAPI *DataGetFromUniqueRecordId)
        (MDS_DB_HANDLE MdsDbHandle,
         const CSSM_DB_UNIQUE_RECORD *UniqueRecord,
         CSSM_DB_RECORD_ATTRIBUTE_DATA_PTR Attributes,
         CSSM_DATA_PTR Data);

    CSSM_RETURN (CSSMAPI *FreeUniqueRecord)
        (MDS_DB_HANDLE MdsDbHandle,
         CSSM_DB_UNIQUE_RECORD_PTR UniqueRecord);

    CSSM_RETURN (CSSMAPI *CreateRelation)
        (MDS_DB_HANDLE MdsDbHandle,
         CSSM_DB_RECORDTYPE RelationID,
         const char *RelationName,
         uint32 NumberOfAttributes,
         const CSSM_DB_SCHEMA_ATTRIBUTE_INFO *pAttributeInfo,
         uint32 NumberOfIndexes,
         const CSSM_DB_SCHEMA_INDEX_INFO *pIndexInfo);

    CSSM_RETURN (CSSMAPI *DestroyRelation)
        (MDS_DB_HANDLE MdsDbHandle,
         CSSM_DB_RECORDTYPE RelationID);
} MDS_FUNCS DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER, *MDS_FUNCS_PTR DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;


/* MDS Context APIs */

CSSM_RETURN CSSMAPI
MDS_Initialize (const CSSM_GUID *pCallerGuid,
                const CSSM_MEMORY_FUNCS *pMemoryFunctions,
                MDS_FUNCS_PTR pDlFunctions,
                MDS_HANDLE *hMds)
				DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

CSSM_RETURN CSSMAPI
MDS_Terminate (MDS_HANDLE MdsHandle)
	DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

CSSM_RETURN CSSMAPI
MDS_Install (MDS_HANDLE MdsHandle)
	DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

CSSM_RETURN CSSMAPI
MDS_Uninstall (MDS_HANDLE MdsHandle)
	DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

#pragma clang diagnostic pop

#ifdef __cplusplus
}
#endif

#endif /* _MDS_H_ */
