/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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
#include "Database.h"
#include "CommonCode.h"



Database::Database (AttachedInstance *ai) : mAttachedInstance (ai)
{
}



Database::~Database ()
{
}



void Database::DbOpen (const char* DbName,
					   const CSSM_NET_ADDRESS *dbLocation,
					   const CSSM_DB_ACCESS_TYPE accessRequest,
					   const CSSM_ACCESS_CREDENTIALS *accessCredentials,
					   const void* openParameters)
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
}



void Database::DbClose ()
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
}



void Database::DbCreate (const char* dbName,
						 const CSSM_NET_ADDRESS *dbLocation,
						 const CSSM_DBINFO *dbInfo,
						 const CSSM_DB_ACCESS_TYPE accessRequest,
						 const CSSM_RESOURCE_CONTROL_CONTEXT *credAndAclEntry,
						 const void *openParameters)
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
}



void Database::DbCreateRelation (CSSM_DB_RECORDTYPE relationID,
								 const char* relationName,
								 uint32 numberOfAttributes,
								 const CSSM_DB_SCHEMA_ATTRIBUTE_INFO *pAttributeInfo,
								 uint32 numberOfIndexes,
								 const CSSM_DB_SCHEMA_INDEX_INFO *pIndexInfo)
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
}



void Database::DbDestroyRelation (CSSM_DB_RECORDTYPE relationID)
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
}



void Database::DbAuthenticate (CSSM_DB_ACCESS_TYPE accessRequest,
							   const CSSM_ACCESS_CREDENTIALS *accessCred)
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
}



void Database::DbGetDbAcl (CSSM_STRING* selectionTag,
						   uint32 *numberOfAclInfos)
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
}



void Database::DbChangeDbAcl (const CSSM_ACCESS_CREDENTIALS *accessCred,
							  const CSSM_ACL_EDIT *aclEdit)
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
}



void Database::DbGetDbOwner (CSSM_ACL_OWNER_PROTOTYPE_PTR owner)
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
}



void Database::DbChangeDbOwner (const CSSM_ACCESS_CREDENTIALS *accessCred,
							    const CSSM_ACL_OWNER_PROTOTYPE *newOwner)
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
}



	
void Database::DbGetDbNameFromHandle (char** dbName)
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
}



void Database::DbDataInsert (CSSM_DB_RECORDTYPE recordType,
							 const CSSM_DB_RECORD_ATTRIBUTE_DATA *attributes,
							 const CSSM_DATA *data,
							 CSSM_DB_UNIQUE_RECORD_PTR *uniqueId)
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
}



void Database::DbDataDelete (const CSSM_DB_UNIQUE_RECORD *uniqueRecordIdentifier)
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
}



void Database::DbDataModify (CSSM_DB_RECORDTYPE recordType,
							 CSSM_DB_UNIQUE_RECORD_PTR uniqueRecordIdentifier,
							 const CSSM_DB_RECORD_ATTRIBUTE_DATA attributesToBeModified,
							 const CSSM_DATA *dataToBeModified,
							 CSSM_DB_MODIFY_MODE modifyMode)
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
}



CSSM_HANDLE Database::DbDataGetFirst (const CSSM_QUERY *query,
									  CSSM_DB_RECORD_ATTRIBUTE_DATA_PTR attributes,
									  CSSM_DATA_PTR data,
									  CSSM_DB_UNIQUE_RECORD_PTR *uniqueID)
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
	
	return 0;
}



bool Database::DbDataGetNext (CSSM_HANDLE resultsHandle,
							  CSSM_DB_RECORD_ATTRIBUTE_DATA_PTR attributes,
							  CSSM_DATA_PTR data,
							  CSSM_DB_UNIQUE_RECORD_PTR *uniqueID)
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
}



void Database::DbDataAbortQuery (CSSM_HANDLE resultsHandle)
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
}



	
void Database::DbDataGetFromUniqueRecordID (const CSSM_DB_UNIQUE_RECORD_PTR uniqueRecord,
										    CSSM_DB_RECORD_ATTRIBUTE_DATA_PTR attributes,
										    CSSM_DATA_PTR data)
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
}



	
void Database::DbFreeUniqueRecord (CSSM_DB_UNIQUE_RECORD_PTR uniqueRecord)
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
}



	
void Database::DbPassThrough (uint32 passThroughID, const void* inputParams, void **outputParams)
{
	CSSMError::ThrowCSSMError (CSSMERR_DL_FUNCTION_NOT_IMPLEMENTED);
}


