/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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
 * ocspdDbSchema.h
 *
 * Definitions of structures which define the schema, including attributes
 * and indexes, for the standard tables that are part of the OCSP server
 * database.
 */

#ifndef _OCSPD_DB_SCHEMA_H_
#define _OCSPD_DB_SCHEMA_H_

#include <Security/cssmtype.h>

/* 
 * Structure used to store information which is needed to create
 * a relation with indexes. The info in one of these structs maps to one
 * record type in a CSSM_DBINFO - both record attribute info and index info.
 */
typedef struct  {
	CSSM_DB_RECORDTYPE				recordType;
	const char						*relationName;
	uint32							numberOfAttributes;
	const CSSM_DB_ATTRIBUTE_INFO	*attrInfo;
	uint32							numIndexes;
	const CSSM_DB_INDEX_INFO		*indexInfo;
} OcspdDbRelationInfo;

// Macros used to simplify declarations of attributes and indexes.

// declare a CSSM_DB_ATTRIBUTE_INFO
#define DB_ATTRIBUTE(name, type) \
	{  CSSM_DB_ATTRIBUTE_NAME_AS_STRING, \
	   { (char*) name }, \
	   CSSM_DB_ATTRIBUTE_FORMAT_ ## type \
	}

// declare a CSSM_DB_INDEX_INFO
#define UNIQUE_INDEX_ATTRIBUTE(name, type) \
	{  CSSM_DB_INDEX_NONUNIQUE, \
	   CSSM_DB_INDEX_ON_ATTRIBUTE, \
	   {  CSSM_DB_ATTRIBUTE_NAME_AS_STRING, \
	      { name }, \
		  CSSM_DB_ATTRIBUTE_FORMAT_ ## type \
	   } \
	}

// declare a OcspdDbRelationInfo
#define RELATION_INFO(relationId, name, attributes, indexes) \
	{ relationId, \
	  name, \
	  sizeof(attributes) / sizeof(CSSM_DB_ATTRIBUTE_INFO), \
	  attributes, \
	  sizeof(indexes) / sizeof(CSSM_DB_INDEX_INFO), \
	  indexes }

/*
 * Currently there is only one relation in the OCSPD database; this is an array
 * containing it. 
 */
extern const OcspdDbRelationInfo kOcspDbRelations[];
extern unsigned kNumOcspDbRelations;

/* 
 * CSSM_DB_RECORDTYPE for the ocspd DB schema.
 */
#define OCSPD_DB_RECORDTYPE		0x11223344

/*
 * Here are the attribute names and formats in kOcspDbRelation. All attributes
 * have format CSSM_DB_ATTRIBUTE_NAME_AS_STRING. 
 */
 
/* DER encoded CertID - a record can have multiple values of this */
#define OCSPD_DBATTR_CERT_ID			DB_ATTRIBUTE("CertID", BLOB)
/* URI */
#define OCSPD_DBATTR_URI				DB_ATTRIBUTE("URI", STRING)
 /* Expiration time, CSSM_TIMESTRING format */
#define OCSPD_DBATTR_EXPIRATION			DB_ATTRIBUTE("Expiration", STRING)

#define OCSPD_NUM_DB_ATTRS		3

#endif /* _OCSPD_DB_SCHEMA_H_ */

