/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 25, 2022.
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
#ifndef _H_SS_TYPES
#define _H_SS_TYPES

#include <sys/syslimits.h>
#include <securityd_client/handletypes.h>

//
// ss_types - common type definitions for securityd-related IPC services
//
#include "ssclient.h"

#define __MigTypeCheck 1


typedef void *Data;
typedef void *XMLBlob;
typedef void *XMLBlobOut;
typedef void *Pointer;
typedef void *BasePointer;

typedef const char *CssmString;

typedef const char *FilePath;
typedef char FilePathOut[PATH_MAX];
typedef void *HashData;
typedef char HashDataOut[maxUcspHashLength];
typedef const char *RelationName;


#ifdef __cplusplus

namespace Security {

using namespace SecurityServer;


// @@@ OBSOLETE BEYOND THIS POINT (SecurityTokend uses this still)

typedef void *ContextAttributes;
typedef Context::Attr *ContextAttributesPointer;

typedef CssmKey *CssmKeyPtr;
typedef AclEntryPrototype *AclEntryPrototypePtr;
typedef AclEntryInput *AclEntryInputPtr;
typedef AclEntryInfo *AclEntryInfoPtr;
typedef AclOwnerPrototype *AclOwnerPrototypePtr;
typedef AccessCredentials *AccessCredentialsPtr;
typedef CssmDeriveData *CssmDeriveDataPtr;

typedef CssmDbRecordAttributeData *CssmDbRecordAttributeDataPtr;
typedef CssmNetAddress *CssmNetAddressPtr;
typedef CssmQuery *CssmQueryPtr;
typedef CssmSubserviceUid *CssmSubserviceUidPtr;
typedef CSSM_DBINFO *CSSM_DBINFOPtr;
typedef CSSM_DB_SCHEMA_ATTRIBUTE_INFO *CSSM_DB_SCHEMA_ATTRIBUTE_INFOPtr;
typedef CSSM_DB_SCHEMA_INDEX_INFO *CSSM_DB_SCHEMA_INDEX_INFOPtr;
typedef CSSM_NAME_LIST *CSSM_NAME_LISTPtr;
typedef void *VoidPtr;

typedef CssmKey::Header CssmKeyHeader;

typedef SecGuestRef *GuestChain;


//
// MIG-used translation functions
//
inline Context &inTrans(CSSM_CONTEXT &arg) { return Context::overlay(arg); }
inline CssmKey &inTrans(CSSM_KEY &arg) { return CssmKey::overlay(arg); }
inline CSSM_KEY &outTrans(CssmKey &key) { return key; }

} // end namespace Security

#endif //__cplusplus


//
// MIG-used byte swapping macros
//
#define __NDR_convert__int_rep__BasePointer__defined
#define __NDR_convert__int_rep__BasePointer(a, f)	/* do not flip */

#endif //_H_SS_TYPES
