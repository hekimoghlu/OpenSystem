/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 30, 2023.
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
#ifndef __DATABASE_H__
#define __DATABASE_H__


#include "AttachedInstance.h"

/*
	Abstract base class for a database -- provides stubs for CSSM calls
*/



class Database
{
protected:
	AttachedInstance *mAttachedInstance;

public:
	Database (AttachedInstance *ai);
	virtual ~Database ();
	
	virtual void DbOpen (const char* DbName,
						 const CSSM_NET_ADDRESS *dbLocation,
						 const CSSM_DB_ACCESS_TYPE accessRequest,
						 const CSSM_ACCESS_CREDENTIALS *accessCredentials,
						 const void* openParameters);
	
	virtual void DbClose ();
	
	virtual void DbCreate (const char* dbName,
						   const CSSM_NET_ADDRESS *dbLocation,
						   const CSSM_DBINFO *dbInfo,
						   const CSSM_DB_ACCESS_TYPE accessRequest,
						   const CSSM_RESOURCE_CONTROL_CONTEXT *credAndAclEntry,
						   const void *openParameters);

	virtual void DbCreateRelation (CSSM_DB_RECORDTYPE relationID,
								   const char* relationName,
								   uint32 numberOfAttributes,
								   const CSSM_DB_SCHEMA_ATTRIBUTE_INFO *pAttributeInfo,
								   uint32 numberOfIndexes,
								   const CSSM_DB_SCHEMA_INDEX_INFO *pIndexInfo);
	
	virtual void DbDestroyRelation (CSSM_DB_RECORDTYPE relationID);

	virtual void DbAuthenticate (CSSM_DB_ACCESS_TYPE accessRequest,
								 const CSSM_ACCESS_CREDENTIALS *accessCred);
	
	virtual void DbGetDbAcl (CSSM_STRING* selectionTag,
							 uint32 *numberOfAclInfos);

	virtual void DbChangeDbAcl (const CSSM_ACCESS_CREDENTIALS *accessCred,
								const CSSM_ACL_EDIT *aclEdit);
	
	virtual void DbGetDbOwner (CSSM_ACL_OWNER_PROTOTYPE_PTR owner);
	
	virtual void DbChangeDbOwner (const CSSM_ACCESS_CREDENTIALS *accessCred,
								  const CSSM_ACL_OWNER_PROTOTYPE *newOwner);
	
	virtual void DbGetDbNameFromHandle (char** dbName);
	
	virtual void DbDataInsert (CSSM_DB_RECORDTYPE recordType,
							   const CSSM_DB_RECORD_ATTRIBUTE_DATA *attributes,
							   const CSSM_DATA *data,
							   CSSM_DB_UNIQUE_RECORD_PTR *uniqueId);
	
	virtual void DbDataDelete (const CSSM_DB_UNIQUE_RECORD *uniqueRecordIdentifier);
	
	virtual void DbDataModify (CSSM_DB_RECORDTYPE recordType,
							   CSSM_DB_UNIQUE_RECORD_PTR uniqueRecordIdentifier,
							   const CSSM_DB_RECORD_ATTRIBUTE_DATA attributesToBeModified,
							   const CSSM_DATA *dataToBeModified,
							   CSSM_DB_MODIFY_MODE modifyMode);
	
	virtual CSSM_HANDLE DbDataGetFirst (const CSSM_QUERY *query,
									    CSSM_DB_RECORD_ATTRIBUTE_DATA_PTR attributes,
									    CSSM_DATA_PTR data,
									    CSSM_DB_UNIQUE_RECORD_PTR *uniqueID);
	
	virtual bool DbDataGetNext (CSSM_HANDLE resultsHandle,
							    CSSM_DB_RECORD_ATTRIBUTE_DATA_PTR attributes,
							    CSSM_DATA_PTR data,
							    CSSM_DB_UNIQUE_RECORD_PTR *uniqueID);
	
	virtual void DbDataAbortQuery (CSSM_HANDLE resultsHandle);
	
	virtual void DbDataGetFromUniqueRecordID (const CSSM_DB_UNIQUE_RECORD_PTR uniqueRecord,
											  CSSM_DB_RECORD_ATTRIBUTE_DATA_PTR attributes,
											  CSSM_DATA_PTR data);
	
	virtual void DbFreeUniqueRecord (CSSM_DB_UNIQUE_RECORD_PTR uniqueRecord);
	
	virtual void DbPassThrough (uint32 passThroughID,
							    const void* inputParams,
							    void **outputParams);
};

#endif
