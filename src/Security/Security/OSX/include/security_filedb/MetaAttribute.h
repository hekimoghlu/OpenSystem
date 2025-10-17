/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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
// MetaAttribute.h
//

#ifndef _H_APPLEDL_METAATTRIBUTE
#define _H_APPLEDL_METAATTRIBUTE

#include "DbValue.h"
#include <memory>

namespace Security
{

// A base class for all meta attributes.

class MetaAttribute
{
public:
	typedef CSSM_DB_ATTRIBUTE_FORMAT Format;
	
	virtual ~MetaAttribute();
	
	// construct an appropriate subclass of MetaAttribute
	static MetaAttribute *create(Format format, uint32 attributeIndex,
		uint32 attributeId);

	Format attributeFormat() const { return mFormat; }
	uint32 attributeIndex() const { return mAttributeIndex; }
	uint32 attributeId() const { return mAttributeId; }

	void packAttribute(WriteSection &ws, uint32 &valueOffset,
		uint32 numValues, const CSSM_DATA *values) const;
	void unpackAttribute(const ReadSection &rs, Allocator &allocator,
		uint32 &numValues, CSSM_DATA *&values) const;
		
	uint32 getNumberOfValues(const ReadSection &rs) const;
	void copyValueBytes(uint32 valueIndex, const ReadSection &rs, WriteSection &ws,
		uint32 &writeOffset) const;

	// interface required of all subclasses, implemented with templates below
	virtual DbValue *createValue(const CSSM_DATA &data) const = 0;
	virtual DbValue *createValue(const ReadSection &rs, uint32 &offset) const = 0;
	virtual void packValue(WriteSection &ws, uint32 &offset, const CSSM_DATA &data) const = 0;
	virtual void unpackValue(const ReadSection &rs, uint32 &offset, CSSM_DATA &data,
		Allocator &allocator) const = 0;
	virtual void skipValue(const ReadSection &rs, uint32 &offset) const = 0;
	virtual void copyValue(const ReadSection &rs, uint32 &readOffset, WriteSection &ws,
		uint32 &writeOffset) const = 0;
	virtual bool evaluate(const DbValue *value, const ReadSection &rs, CSSM_DB_OPERATOR op) const = 0;
	virtual bool evaluate(const DbValue *value1, const DbValue *value2, CSSM_DB_OPERATOR op) const = 0;
	virtual uint32 parse(const CssmData &inData, CSSM_DATA_PTR &outValues) const = 0;

protected:
	MetaAttribute(Format format, uint32 attributeIndex, uint32 attributeId)
	: mFormat(format), mAttributeIndex(attributeIndex), mAttributeId(attributeId) {}

	void packNumberOfValues(WriteSection &ws, uint32 numValues, uint32 &valueOffset) const;
	void unpackNumberOfValues(const ReadSection &rs, uint32 &numValues, uint32 &valueOffset) const;

	Format mFormat;
	uint32 mAttributeIndex;
	uint32 mAttributeId;
};

// Template used to describe particular subclasses of MetaAttribute

template <class T>
class TypedMetaAttribute : public MetaAttribute
{
public:
	TypedMetaAttribute(Format format, uint32 attributeIndex, uint32 attributeId)
	: MetaAttribute(format, attributeIndex, attributeId) {}

	DbValue *createValue(const CSSM_DATA &data) const
	{
		return new T(data);
	}

	DbValue *createValue(const ReadSection &rs, uint32 &offset) const
	{
		return new T(rs, offset);
	}

	void packValue(WriteSection &ws, uint32 &offset, const CSSM_DATA &data) const
	{
		T value(data);
		value.pack(ws, offset);
	}

	void unpackValue(const ReadSection &rs, uint32 &offset, CSSM_DATA &data, Allocator &allocator) const
	{
		T value(rs, offset);
		data.Length = value.size();

		if (data.Length != 0)
		{
			data.Data = reinterpret_cast<uint8 *>(allocator.malloc(data.Length));
			memcpy(data.Data, value.bytes(), data.Length);
		}
		else
		{
			data.Data = NULL;
		}
	}

	void skipValue(const ReadSection &rs, uint32 &offset) const
	{
		T value(rs, offset);
	}
	
	void copyValue(const ReadSection &rs, uint32 &readOffset, WriteSection &ws, uint32 &writeOffset) const
	{
		T value(rs, readOffset);
		value.pack(ws, writeOffset);
	}

	bool evaluate(const DbValue *value, const ReadSection &rs, CSSM_DB_OPERATOR op) const
	{
		uint32 offset, numValues;
		unpackNumberOfValues(rs, numValues, offset);

		/* If any of the values for this attribute match we have a
		   match.   This is the same behaviour that indexes have. */
		for (uint32 ix = 0; ix < numValues; ++ix)
			if (dynamic_cast<const T *>(value)->evaluate(T(rs, offset), op))
				return true;

		return false;
	}

	bool evaluate(const DbValue *value1, const DbValue *value2, CSSM_DB_OPERATOR op) const
	{
		return (dynamic_cast<const T *>(value1))->evaluate(*dynamic_cast<const T *>(value2), op);
	}
		
    uint32 parse(const CssmData &inData, CSSM_DATA_PTR &outValues) const
	{
		return 0;
	}
};

} // end namespace Security

#endif //  _H_APPLEDL_METAATTRIBUTE
