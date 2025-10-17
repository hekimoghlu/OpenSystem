/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 30, 2025.
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
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include "TableRelation.h"
#include "CommonCode.h"

struct TableRelationStruct
{
	CSSM_DB_RECORDTYPE recordType;
};



void TableTuple::GetData (CSSM_DATA &data)
{
	data.Data = NULL;
	data.Length = 0;
}



void TableUniqueIdentifier::Export (CSSM_DB_UNIQUE_RECORD &record)
{
	// we don't care about any of the fields of this record, so just zero it out.
	memset (&record, 0, sizeof (record));
}



TableUniqueIdentifier::TableUniqueIdentifier (CSSM_DB_RECORDTYPE recordType, int tupleNumber) : UniqueIdentifier (recordType), mTupleNumber (tupleNumber)
{
}



struct TableUniqueIdentifierStruct : public TableRelationStruct
{
	int tupleNumber;
};



TableRelation::TableRelation (CSSM_DB_RECORDTYPE recordType, int numberOfColumns, columnInfoLoader *theColumnInfo) : PartialRelation (recordType, numberOfColumns, theColumnInfo), mNumberOfTuples (0), mData (NULL)
{
}



TableRelation::~TableRelation ()
{
	if (mData != NULL)
	{
		int arraySize = mNumberOfTuples * mNumberOfColumns;
		int i;
		for (i = 0; i < arraySize; ++i)
		{
			delete mData[i];
		}
		
		free (mData);
	}
}



void TableRelation::AddTuple (Value* column0Value, ...)
{
	// extend the tuple array by the number of tuple to be added
	int n = mNumberOfTuples++ * mNumberOfColumns;
	int newArraySize = n + mNumberOfColumns;
	mData = (Value**) realloc (mData, newArraySize * sizeof (Value*));
	
	mData[n++] = column0Value;
	
	va_list argList;
	va_start (argList, column0Value);

	int i;
	for (i = 1; i < mNumberOfColumns; ++i)
	{
		Value* next = va_arg (argList, Value*);
		mData[n++] = next;
	}
	
	va_end (argList);
}



Query* TableRelation::MakeQuery (const CSSM_QUERY* query)
{
	return new TableQuery (this, query);
}



Tuple* TableRelation::GetTuple (int i)
{
	Value** offset = mData + i * mNumberOfColumns;
	TableTuple* tt = new TableTuple (offset, mNumberOfColumns);
	return tt;
}



Tuple* TableRelation::GetTupleFromUniqueIdentifier (UniqueIdentifier* uniqueID)
{
	TableUniqueIdentifier *id = (TableUniqueIdentifier*) uniqueID;
	return GetTuple (id->GetTupleNumber ());
}



UniqueIdentifier* TableRelation::ImportUniqueIdentifier (CSSM_DB_UNIQUE_RECORD *uniqueRecord)
{
	TableUniqueIdentifierStruct *is = (TableUniqueIdentifierStruct *) uniqueRecord->RecordIdentifier.Data;
	TableUniqueIdentifier* it = new TableUniqueIdentifier (is->recordType, is->tupleNumber);
	return it;
}



TableTuple::TableTuple (Value** offset, int numValues) : mValues (offset), mNumValues (numValues)
{
}



TableQuery::TableQuery (TableRelation* relation, const CSSM_QUERY *queryBase) : Query (relation, queryBase), mRelation (relation), mCurrentRecord (0)
{
}



TableQuery::~TableQuery ()
{
}



Tuple* TableQuery::GetNextTuple (UniqueIdentifier *& id)
{
	while (mCurrentRecord < mRelation->GetNumberOfTuples ()) {
		Tuple *t = mRelation->GetTuple (mCurrentRecord);

		if (EvaluateTuple (t)) {
			id = new TableUniqueIdentifier (mRelation->GetRecordType (), mCurrentRecord);
			mCurrentRecord += 1;
			return t;
		}
		mCurrentRecord += 1;
		delete t;
	}
	
	id = NULL;
	return NULL;
}
