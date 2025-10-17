/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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
#ifndef __DSX509RELATION__
#define __DSX509RELATION__

#include "PartialRelation.h"
// #include "DirectoryServices.h"
#include "ODBridge.h"


/*
	These classes define the relationship between CDSA and Open Directory
*/

// relation column numbers
enum {kCertTypeID = 0, kCertEncodingID, kCertPrintName, kCertAlias, kCertSubject, kCertIssuer, kCertSerialNumber,
	  kCertSubjectKeyIdentifier, kCertPublicKeyHash};

const int kNumberOfX509Attributes = kCertPublicKeyHash - kCertTypeID + 1;

// the "tuple" we return
class DSX509Tuple : public Tuple
{
protected:
	int mNumberOfValues;							// number of attributes
	Value** mValues;								// the attributes themselves
	BlobValue *mData;								// the data for this tuple

public:
	DSX509Tuple (int numberOfValues);
	virtual ~DSX509Tuple ();

	void SetValue (int i, Value* v);				// set an attribute by column number
	
	Value* GetValue (int i);						// get an attribute
	
	int GetNumberOfValues ();						// number of attributes
	
	void GetData (CSSM_DATA &data);					// get the data
	void SetData (BlobValue *value);				// set the data
};



class DSX509Relation;

// a class representing a single open directory record, and the method to serialize it as a tuple
class DSX509Record
{
protected:
	DSX509Relation *mRelation;

public:
	DSX509Record (DSX509Relation* relation) : mRelation (relation) {}
	DSX509Tuple* GetTuple (CFDataRef certData, CFStringRef original_search, DSX509Tuple *tupleList[], int maxTuples);
};


// a class representing a unique identifier for a record (in the CDSA sense)
class DSX509UniqueIdentifier : public UniqueIdentifier
{
protected:
	DSX509Tuple *mTuple;

public:
	DSX509UniqueIdentifier (DSX509Tuple *t);
	virtual ~DSX509UniqueIdentifier ();
	virtual void Export (CSSM_DB_UNIQUE_RECORD &record);
	DSX509Tuple* GetTuple ();
};



const int kMaxTuples = 10;

// a class which converts between a CDSA query and an open directory lookup
class DSX509Query : public Query
{
protected:
	DirectoryService *mDirectoryService;								// the directory service instance from which we came
	// DSContext *mDSContext;											// our current context
	unsigned long mRecordCount;											// the record we are currently searching
	unsigned long mCurrentItem;											// the item we are currently searching
	CSSM_QUERY *queryBase;												// The original query
	ODdl_results_handle mRecordList;									// the records we are searching
	bool validQuery;
	bool ValidateQueryString(CSSM_DATA mailAddr);
	Tuple* MakeTupleFromRecord (CFDataRef record);						// convert a record to a tuple

	DSX509Tuple* mTupleList[kMaxTuples];								// store tuples returned from a query
	int mNumberOfTuples;												// number of tuples stored
	int mNextTuple;														// next tuple to be returned

public:
	DSX509Query (DSX509Relation* relation, const CSSM_QUERY *queryBase);
	virtual ~DSX509Query ();
	
	virtual Tuple* GetNextTuple (UniqueIdentifier *&id);				// get a tuple and return an ID that identifies it
};



class DSX509Relation : public PartialRelation
{
protected:
	CSSM_CL_HANDLE mCertificateLibrary;
	
	void InitializeCertLibrary ();										// load the CL

public:
	DirectoryService *mDirectoryService;
	
	DSX509Relation (CSSM_DB_RECORDTYPE recordType, int numberOfColumns, columnInfoLoader *theColumnInfo);
	virtual ~DSX509Relation ();

	Query* MakeQuery (const CSSM_QUERY* query);							// convert a CSSM_QUERY object to an internal form
	Tuple* GetTupleFromUniqueIdentifier (UniqueIdentifier* uniqueID);	// get tuple by unique ID
	UniqueIdentifier* ImportUniqueIdentifier (CSSM_DB_UNIQUE_RECORD *uniqueRecord);	// make a unique ID from an external form
	CSSM_CL_HANDLE GetCLHandle ();										// get the CL handle -- initialize if necessary
};



#endif
