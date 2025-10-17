/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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
 * ocspdDbSchema.cpp
 *
 * Definitions of structures which define the schema, including attributes
 * and indexes, for the standard tables that are part of the OCSP server
 * database.
 */

#include "ocspdDbSchema.h"
#include <cstring>

//
// Schema for the lone table in the OCSPD database.
//
static const CSSM_DB_ATTRIBUTE_INFO ocspdDbAttrs[] = {
	OCSPD_DBATTR_CERT_ID,
	OCSPD_DBATTR_URI,
	OCSPD_DBATTR_EXPIRATION
};

static const CSSM_DB_INDEX_INFO ocspdDbIndex[] = {
	UNIQUE_INDEX_ATTRIBUTE((char*) "CertID", BLOB)
};

const OcspdDbRelationInfo kOcspDbRelations[] =
{
	RELATION_INFO(OCSPD_DB_RECORDTYPE, "ocpsd", ocspdDbAttrs, ocspdDbIndex)
};

unsigned kNumOcspDbRelations = sizeof(kOcspDbRelations) / sizeof(kOcspDbRelations[0]);

