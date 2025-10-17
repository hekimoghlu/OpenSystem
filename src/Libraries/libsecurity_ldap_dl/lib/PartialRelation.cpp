/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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
#include "PartialRelation.h"
#include "TableRelation.h"
#include "CommonCode.h"


PartialRelation::PartialRelation (CSSM_DB_RECORDTYPE recordType, int numberOfColumns, columnInfoLoader *theColumnInfo)  :
	Relation (recordType), mNumberOfColumns (numberOfColumns)
{
	if (mNumberOfColumns == 0) {
		mColumnInfo = NULL;
		return;
	}
	
	mColumnInfo = new columnInfo[mNumberOfColumns];
	for (int i = 0; i < mNumberOfColumns; ++i) {
		mColumnInfo[i].mColumnName = new StringValue (theColumnInfo[i].mColumnName);
		mColumnInfo[i].mColumnID = theColumnInfo[i].mColumnID;
		mColumnInfo[i].mColumnFormat = theColumnInfo[i].mColumnFormat;
	}
}



PartialRelation::~PartialRelation ()
{
	if (mColumnInfo != NULL) {
		for (int i = 0; i < mNumberOfColumns; ++i)
			delete mColumnInfo[i].mColumnName;		
		delete mColumnInfo;
	}
}


StringValue *PartialRelation::GetColumnName (int i)
{
	return mColumnInfo[i].mColumnName;
}


int PartialRelation::GetNumberOfColumns ()
{
	return mNumberOfColumns;
}



uint32 PartialRelation::GetColumnIDs (int i)
{
	return mColumnInfo[i].mColumnID;
}



int PartialRelation::GetColumnNumber (const char* columnName)
{
	// look for a column name that matches this columnName.  If not, throw an exception
 	for (int i = 0; i < mNumberOfColumns; ++i) {
		const char *s = mColumnInfo[i].mColumnName->GetRawValue();
		if (strncmp(s, columnName, strlen(s)) == 0)
 			return i;
 	}
	return -1;
}



int PartialRelation::GetColumnNumber (uint32 columnID)
{
	for (int i = 0; i < mNumberOfColumns; ++i) {
		if (mColumnInfo[i].mColumnID == columnID)
			return i;
	}
	return -1;
}



