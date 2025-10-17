/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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
#ifndef __TABLE_RELATION__
#define __TABLE_RELATION__


#include "PartialRelation.h"

/* 
	a table relation is a relation which is completely stored in memory -- used for
	indexes, lists of relations, lists of attributes, etc.
*/

// a unique identifier for a table
class TableUniqueIdentifier : public UniqueIdentifier
{
protected:
	uint32 mTupleNumber;

public:
	TableUniqueIdentifier (CSSM_DB_RECORDTYPE recordType, int mTupleNumber);
	virtual void Export (CSSM_DB_UNIQUE_RECORD &record);
	
	int GetTupleNumber () {return mTupleNumber;}
};



// a table relation.  Uses PartialRelation to track basic info.
class TableRelation : public PartialRelation
{
protected:
	int mNumberOfTuples;
	Value** mData;

public:
	TableRelation (CSSM_DB_RECORDTYPE recordType, int numberOfColumns, columnInfoLoader *theColumnInfo);
	virtual ~TableRelation ();
	
	void AddTuple (Value* column0Value, ...);

	virtual Query* MakeQuery (const CSSM_QUERY* query);
	virtual Tuple* GetTupleFromUniqueIdentifier (UniqueIdentifier* uniqueID);
	virtual Tuple* GetTuple (int i);
	virtual UniqueIdentifier* ImportUniqueIdentifier (CSSM_DB_UNIQUE_RECORD *uniqueRecord);
	int GetNumberOfTuples () {return mNumberOfTuples;}
};



// a tuple for a TableRelation
class TableTuple : public Tuple
{
protected:
	Value** mValues;
	int mNumValues;

public:
	TableTuple (Value** offset, int numValues);
	virtual int GetNumberOfValues () {return mNumValues;}
	virtual Value* GetValue (int which) {return mValues[which];}
	virtual void GetData (CSSM_DATA &data);
};



// a query for a TableRelation
class TableQuery : public Query
{
protected:
	TableRelation* mRelation;
	int mCurrentRecord;

public:
	TableQuery (TableRelation* relation, const CSSM_QUERY *queryBase);
	~TableQuery ();
	
	virtual Tuple* GetNextTuple (UniqueIdentifier *&id);
};



#endif
