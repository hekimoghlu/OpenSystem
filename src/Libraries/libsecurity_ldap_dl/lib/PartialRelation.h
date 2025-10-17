/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 26, 2022.
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
#ifndef __PARTIAL_RELATION__
#define __PARTIAL_RELATION__



#include "Relation.h"

typedef struct columnInfoLoader {
	uint32						mColumnID;
	const char					*mColumnName;
	CSSM_DB_ATTRIBUTE_FORMAT	mColumnFormat;
} columnInfoLoader;

typedef struct columnInfo {
	uint32						mColumnID;
	StringValue					*mColumnName;
	CSSM_DB_ATTRIBUTE_FORMAT	mColumnFormat;
} columnInfo;

/*
	PartialRelation.h
	
	This class provides common support for writing relations.
*/

class PartialRelation : public Relation
{
protected:
	int mNumberOfColumns;												// number of columns (attributes) this relation supports
	columnInfo *mColumnInfo;

public:
	PartialRelation (CSSM_DB_RECORDTYPE recordType, int numberOfColumns, columnInfoLoader *theColumnInfo);
																		// pass in the relation ID and number of columns
	virtual ~PartialRelation ();

	virtual StringValue *GetColumnName (int i);									// returns an array of Tuples representing the column names
	virtual int GetNumberOfColumns ();									// returns the number of columns
	int GetColumnNumber (const char* columnName);						// returns the column number corresponding to the name
	int GetColumnNumber (uint32 columnID);								// returns the column number corresponding to the ID
	
	CSSM_DB_ATTRIBUTE_FORMAT GetColumnFormat (int i) {return mColumnInfo[i].mColumnFormat;} // returns the format of a column
	uint32 GetColumnIDs (int i);											// gets the column id's
};



#endif
