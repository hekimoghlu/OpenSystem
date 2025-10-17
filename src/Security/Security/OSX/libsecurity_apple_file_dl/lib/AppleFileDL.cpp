/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 22, 2022.
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
// AppleFileDL.cpp - File Based DL plug-in module.
//
#include "AppleFileDL.h"

#include <security_cdsa_plugin/DLsession.h>


// Names and IDs of tables used in a DL database

static const AppleDatabaseTableName kTableNames[] = {
	{ CSSM_DL_DB_SCHEMA_INFO, "CSSM_DL_DB_SCHEMA_INFO" },
	{ CSSM_DL_DB_SCHEMA_ATTRIBUTES, "CSSM_DL_DB_SCHEMA_ATTRIBUTES" },
	{ CSSM_DL_DB_SCHEMA_INDEXES, "CSSM_DL_DB_SCHEMA_INDEXES" },
	{ CSSM_DL_DB_SCHEMA_PARSING_MODULE, "CSSM_DL_DB_SCHEMA_PARSING_MODULE" },
	{ CSSM_DL_DB_RECORD_CERT, "CSSM_DL_DB_RECORD_CERT" },
	{ CSSM_DL_DB_RECORD_CRL, "CSSM_DL_DB_RECORD_CRL" },
	{ CSSM_DL_DB_RECORD_POLICY, "CSSM_DL_DB_RECORD_POLICY" },
	{ CSSM_DL_DB_RECORD_GENERIC, "CSSM_DL_DB_RECORD_GENERIC" },
	{ CSSM_DL_DB_RECORD_PUBLIC_KEY, "CSSM_DL_DB_RECORD_PUBLIC_KEY" },
	{ CSSM_DL_DB_RECORD_PRIVATE_KEY, "CSSM_DL_DB_RECORD_PRIVATE_KEY" },
	{ CSSM_DL_DB_RECORD_SYMMETRIC_KEY, "CSSM_DL_DB_RECORD_SYMMETRIC_KEY" },
	{ ~0U, NULL }
};

//
// Make and break the plugin object
//
AppleFileDL::AppleFileDL()
	:	mDatabaseManager(kTableNames)
{
}

AppleFileDL::~AppleFileDL()
{
}


//
// Create a new plugin session, our way
//
PluginSession *AppleFileDL::makeSession(CSSM_MODULE_HANDLE handle,
                                       const CSSM_VERSION &version,
                                       uint32 subserviceId,
                                       CSSM_SERVICE_TYPE subserviceType,
                                       CSSM_ATTACH_FLAGS attachFlags,
                                       const CSSM_UPCALLS &upcalls)
{
    switch (subserviceType) {
        case CSSM_SERVICE_DL:
            return new DLPluginSession(handle,
                                       *this,
                                       version,
                                       subserviceId,
                                       subserviceType,
                                       attachFlags,
                                       upcalls,
                                       mDatabaseManager);
        default:
            CssmError::throwMe(CSSMERR_CSSM_INVALID_SERVICE_MASK);
    }
}
