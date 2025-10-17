/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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
// SDDLSession.h - DL session for security server CSP/DL.
//
#ifndef _H_SDDLSESSION
#define _H_SDDLSESSION

#include <security_cdsa_plugin/DLsession.h>
#include <security_cdsa_utilities/u32handleobject.h>
#include <securityd_client/ssclient.h>

class SDCSPDLPlugin;
class SDCSPDLSession;

class SDDLSession : public DLPluginSession
{
public:
	SDCSPDLSession &mSDCSPDLSession;

	SDDLSession(CSSM_MODULE_HANDLE handle,
				SDCSPDLPlugin &plug,
				const CSSM_VERSION &version,
				uint32 subserviceId,
				CSSM_SERVICE_TYPE subserviceType,
				CSSM_ATTACH_FLAGS attachFlags,
				const CSSM_UPCALLS &upcalls,
				DatabaseManager &databaseManager,
				SDCSPDLSession &ssCSPDLSession);
	~SDDLSession();

	SecurityServer::ClientSession &clientSession()
	{ return mClientSession; }
    void GetDbNames(CSSM_NAME_LIST_PTR &NameList);
    void FreeNameList(CSSM_NAME_LIST &NameList);
    void DbDelete(const char *DbName,
                  const CSSM_NET_ADDRESS *DbLocation,
                  const AccessCredentials *AccessCred);
    void DbCreate(const char *DbName,
                  const CSSM_NET_ADDRESS *DbLocation,
                  const CSSM_DBINFO &DBInfo,
                  CSSM_DB_ACCESS_TYPE AccessRequest,
                  const CSSM_RESOURCE_CONTROL_CONTEXT *CredAndAclEntry,
                  const void *OpenParameters,
                  CSSM_DB_HANDLE &DbHandle);
    void DbOpen(const char *DbName,
                const CSSM_NET_ADDRESS *DbLocation,
                CSSM_DB_ACCESS_TYPE AccessRequest,
                const AccessCredentials *AccessCred,
                const void *OpenParameters,
                CSSM_DB_HANDLE &DbHandle);
    void DbClose(CSSM_DB_HANDLE DBHandle);
    void CreateRelation(CSSM_DB_HANDLE DBHandle,
                        CSSM_DB_RECORDTYPE RelationID,
                        const char *RelationName,
                        uint32 NumberOfAttributes,
                        const CSSM_DB_SCHEMA_ATTRIBUTE_INFO *pAttributeInfo,
                        uint32 NumberOfIndexes,
                        const CSSM_DB_SCHEMA_INDEX_INFO &pIndexInfo);
    void DestroyRelation(CSSM_DB_HANDLE DBHandle,
                         CSSM_DB_RECORDTYPE RelationID);

    void Authenticate(CSSM_DB_HANDLE DBHandle,
                      CSSM_DB_ACCESS_TYPE AccessRequest,
                      const AccessCredentials &AccessCred);
    void GetDbAcl(CSSM_DB_HANDLE DBHandle,
                  const CSSM_STRING *SelectionTag,
                  uint32 &NumberOfAclInfos,
                  CSSM_ACL_ENTRY_INFO_PTR &AclInfos);
    void ChangeDbAcl(CSSM_DB_HANDLE DBHandle,
                     const AccessCredentials &AccessCred,
                     const CSSM_ACL_EDIT &AclEdit);
    void GetDbOwner(CSSM_DB_HANDLE DBHandle,
                    CSSM_ACL_OWNER_PROTOTYPE &Owner);
    void ChangeDbOwner(CSSM_DB_HANDLE DBHandle,
                       const AccessCredentials &AccessCred,
                       const CSSM_ACL_OWNER_PROTOTYPE &NewOwner);
    void GetDbNameFromHandle(CSSM_DB_HANDLE DBHandle,
                             char **DbName);
    void DataInsert(CSSM_DB_HANDLE DBHandle,
                    CSSM_DB_RECORDTYPE RecordType,
                    const CSSM_DB_RECORD_ATTRIBUTE_DATA *Attributes,
                    const CssmData *Data,
                    CSSM_DB_UNIQUE_RECORD_PTR &UniqueId);
    void DataDelete(CSSM_DB_HANDLE DBHandle,
                    const CSSM_DB_UNIQUE_RECORD &UniqueRecordIdentifier);
    void DataModify(CSSM_DB_HANDLE DBHandle,
                    CSSM_DB_RECORDTYPE RecordType,
                    CSSM_DB_UNIQUE_RECORD &UniqueRecordIdentifier,
                    const CSSM_DB_RECORD_ATTRIBUTE_DATA *AttributesToBeModified,
                    const CssmData *DataToBeModified,
                    CSSM_DB_MODIFY_MODE ModifyMode);
    CSSM_HANDLE DataGetFirst(CSSM_DB_HANDLE DBHandle,
                             const CssmQuery *Query,
                             CSSM_DB_RECORD_ATTRIBUTE_DATA_PTR Attributes,
                             CssmData *Data,
                             CSSM_DB_UNIQUE_RECORD_PTR &UniqueId);
    bool DataGetNext(CSSM_DB_HANDLE DBHandle,
                     CSSM_HANDLE ResultsHandle,
                     CSSM_DB_RECORD_ATTRIBUTE_DATA_PTR Attributes,
                     CssmData *Data,
                     CSSM_DB_UNIQUE_RECORD_PTR &UniqueId);
    void DataAbortQuery(CSSM_DB_HANDLE DBHandle,
                        CSSM_HANDLE ResultsHandle);
    void DataGetFromUniqueRecordId(CSSM_DB_HANDLE DBHandle,
                                   const CSSM_DB_UNIQUE_RECORD &UniqueRecord,
                                   CSSM_DB_RECORD_ATTRIBUTE_DATA_PTR Attributes,
                                   CssmData *Data);
    void FreeUniqueRecord(CSSM_DB_HANDLE DBHandle,
                          CSSM_DB_UNIQUE_RECORD &UniqueRecord);
    void PassThrough(CSSM_DB_HANDLE DBHandle,
                     uint32 PassThroughId,
                     const void *InputParams,
                     void **OutputParams);

	Allocator &allocator() { return *static_cast<DatabaseSession *>(this); }

protected:
	void postGetRecord(SecurityServer::RecordHandle record, U32HandleObject::Handle resultsHandle,
					   CSSM_DB_HANDLE db,
					   CssmDbRecordAttributeData *pAttributes,
					   CSSM_DB_RECORD_ATTRIBUTE_DATA_PTR inoutAttributes,
					   CssmData *inoutData, SecurityServer::KeyHandle hKey);

	CSSM_DB_UNIQUE_RECORD_PTR makeDbUniqueRecord(SecurityServer::RecordHandle recordHandle);
	CSSM_HANDLE findDbUniqueRecord(const CSSM_DB_UNIQUE_RECORD &inUniqueRecord);
	void freeDbUniqueRecord(CSSM_DB_UNIQUE_RECORD &inUniqueRecord);

	SecurityServer::ClientSession mClientSession;
    //SecurityServer::AttachmentHandle mAttachment;
};


#endif // _H_SDDLSESSION
