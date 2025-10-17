/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 27, 2024.
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
// DbValue.h
//

#ifndef _H_APPLEDL_DBVALUE
#define _H_APPLEDL_DBVALUE

#include "ReadWriteSection.h"

#include <security_cdsa_utilities/cssmdata.h>
#include <security_cdsa_utilities/cssmdb.h>
#include <Security/cssmerr.h>
#include <map>
#include <vector>

namespace Security
{

//
// DbValue -- A base class for all types of database values.
//
class DbValue
{
public:
	virtual ~DbValue();
};

// A collection of subclasses of DbValue that work for simple
// data types, e.g. uint32, sint32, and double, that have
// the usual C comparison and sizeof operations. Defining this
// template saves typing below.

template <class T>
class BasicValue : public DbValue
{
public:
	BasicValue() {}
	BasicValue(T value) : mValue(value) {}

	bool evaluate(const BasicValue<T> &other, CSSM_DB_OPERATOR op) const
	{
		switch (op) {

		case CSSM_DB_EQUAL:
			return mValue == other.mValue;
			
		case CSSM_DB_NOT_EQUAL:
			return mValue != other.mValue;
			
		case CSSM_DB_LESS_THAN:
			return mValue < other.mValue;

		case CSSM_DB_GREATER_THAN:
			return mValue > other.mValue;			

		default:
			CssmError::throwMe(CSSMERR_DL_UNSUPPORTED_QUERY);
			return false;
		}
	}

	size_t size() const { return sizeof(T); }
	size_t size(const ReadSection &rs, uint32 offset) const { return size(); }
	const uint8 *bytes() const { return reinterpret_cast<const uint8 *>(&mValue); }

protected:
	T mValue;
};

// Actual useful subclasses of DbValue as instances of BasicValue.
// Note that all of these require a constructor of the form
// (const ReadSection &, uint32 &offset) that advances the offset
// to just after the value.

class UInt32Value : public BasicValue<uint32>
{
public:
	UInt32Value(const ReadSection &rs, uint32 &offset);		
	UInt32Value(const CSSM_DATA &data);
	virtual ~UInt32Value();
	void pack(WriteSection &ws, uint32 &offset) const;
};

class SInt32Value : public BasicValue<sint32>
{
public:
	SInt32Value(const ReadSection &rs, uint32 &offset);		
	SInt32Value(const CSSM_DATA &data);
	virtual ~SInt32Value();
	void pack(WriteSection &ws, uint32 &offset) const;
};

class DoubleValue : public BasicValue<double>
{
public:
	DoubleValue(const ReadSection &rs, uint32 &offset);		
	DoubleValue(const CSSM_DATA &data);
	virtual ~DoubleValue();
	void pack(WriteSection &ws, uint32 &offset) const;
};

// Subclasses of Value for more complex types.

class BlobValue : public DbValue, public CssmData
{
public:
	BlobValue() {}
	BlobValue(const ReadSection &rs, uint32 &offset);		
	BlobValue(const CSSM_DATA &data);
	virtual ~BlobValue();
	void pack(WriteSection &ws, uint32 &offset) const;
	bool evaluate(const BlobValue &other, CSSM_DB_OPERATOR op) const;

	size_t size() const { return Length; }
	const uint8 *bytes() const { return Data; }
	
protected:
	class Comparator {
	public:
		virtual ~Comparator();
		virtual int operator () (const uint8 *ptr1, const uint8 *ptr2, uint32 length);
	};

	static bool evaluate(const CssmData &data1, const CssmData &data2, CSSM_DB_OPERATOR op,
		Comparator compare);
};

class TimeDateValue : public BlobValue
{
public:
	enum { kTimeDateSize = 16 };

	TimeDateValue(const ReadSection &rs, uint32 &offset);		
	TimeDateValue(const CSSM_DATA &data);
	virtual ~TimeDateValue();
	void pack(WriteSection &ws, uint32 &offset) const;

	bool isValidDate() const;
	
private:
	uint32 rangeValue(uint32 start, uint32 length) const;
};

class StringValue : public BlobValue
{
public:
	StringValue(const ReadSection &rs, uint32 &offset);		
	StringValue(const CSSM_DATA &data);
	virtual ~StringValue();
	bool evaluate(const StringValue &other, CSSM_DB_OPERATOR op) const;
	
private:
	class Comparator : public BlobValue::Comparator {
	public:
		virtual int operator () (const uint8 *ptr1, const uint8 *ptr2, uint32 length);
	};

};

class BigNumValue : public BlobValue
{
public:
	static const uint8 kSignBit = 0x80;

	BigNumValue(const ReadSection &rs, uint32 &offset);		
	BigNumValue(const CSSM_DATA &data);
	virtual ~BigNumValue();
	bool evaluate(const BigNumValue &other, CSSM_DB_OPERATOR op) const;

private:
	static int compare(const uint8 *a, const uint8 *b, int length);
};

class MultiUInt32Value : public DbValue
{
public:
	MultiUInt32Value(const ReadSection &rs, uint32 &offset);		
	MultiUInt32Value(const CSSM_DATA &data);
	virtual ~MultiUInt32Value();
	void pack(WriteSection &ws, uint32 &offset) const;
	bool evaluate(const MultiUInt32Value &other, CSSM_DB_OPERATOR op) const;

	size_t size() const { return mNumValues * sizeof(uint32); }
	const uint8 *bytes() const { return reinterpret_cast<uint8 *>(mValues); }
	
private:
	uint32 mNumValues;
	uint32 *mValues;
	bool mOwnsValues;
};

} // end namespace Security

#endif // _H_APPLEDL_DBVALUE

