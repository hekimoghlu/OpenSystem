/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 5, 2023.
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
// dlquery - search query sublanguage for DL and MDS queries
//
#include <security_cdsa_client/dlquery.h>


namespace Security {
namespace CssmClient {


//
// Constructing Relations
//
Comparison::Comparison(const Comparison &r)
	: mName(r.mName), mOperator(r.mOperator), mFormat(r.mFormat),
	  mValue(Allocator::standard())
{
	mValue.copy(r.mValue);
}
	
Comparison &Comparison::operator = (const Comparison &r)
{
	mName = r.mName;
	mOperator = r.mOperator;
	mFormat = r.mFormat;
	mValue.copy(r.mValue);
	return *this;
}


Comparison::Comparison(const Attribute &attr, CSSM_DB_OPERATOR op, const char *s)
	: mName(attr.name()), mOperator(op), mFormat(CSSM_DB_ATTRIBUTE_FORMAT_STRING),
	mValue(Allocator::standard(), StringData(s))
{ }

Comparison::Comparison(const Attribute &attr, CSSM_DB_OPERATOR op, const std::string &s)
	: mName(attr.name()), mOperator(op), mFormat(CSSM_DB_ATTRIBUTE_FORMAT_STRING),
	mValue(Allocator::standard(), StringData(s))
{ }

Comparison::Comparison(const Attribute &attr, CSSM_DB_OPERATOR op, uint32 value)
	: mName(attr.name()), mOperator(op), mFormat(CSSM_DB_ATTRIBUTE_FORMAT_UINT32),
	mValue(Allocator::standard(), CssmData::wrap(value))
{ }

Comparison::Comparison(const Attribute &attr, CSSM_DB_OPERATOR op, bool value)
	: mName(attr.name()), mOperator(op), mFormat(CSSM_DB_ATTRIBUTE_FORMAT_UINT32),
	mValue(Allocator::standard(), CssmData::wrap(uint32(value ? 1 : 0)))
{ }

Comparison::Comparison(const Attribute &attr, CSSM_DB_OPERATOR op, const CssmData &data)
	: mName(attr.name()), mOperator(op), mFormat(CSSM_DB_ATTRIBUTE_FORMAT_BLOB),
	mValue(Allocator::standard(), data)
{ }

Comparison::Comparison(const Attribute &attr, CSSM_DB_OPERATOR op, const CSSM_GUID &guid)
	: mName(attr.name()), mOperator(op), mFormat(CSSM_DB_ATTRIBUTE_FORMAT_STRING),
	mValue(Allocator::standard(), StringData(Guid::overlay(guid).toString()))
{
}


Comparison::Comparison(const Attribute &attr)
	: mName(attr.name()), mOperator(CSSM_DB_NOT_EQUAL), mFormat(CSSM_DB_ATTRIBUTE_FORMAT_UINT32),
	  mValue(Allocator::standard(), CssmData::wrap(uint32(CSSM_FALSE)))
{
}

Comparison operator ! (const Attribute &attr)
{
	return Comparison(attr, CSSM_DB_EQUAL, uint32(CSSM_FALSE));
}


//
// Query methods
//
Query &Query::operator = (const Query &q)
{
	mRelations = q.mRelations;
	mQueryValid = false;
	return *this;
}


//
// Form the CssmQuery from a Query object.
// We cache this in mQuery, which we have made sure isn't copied along.
//
const CssmQuery &Query::cssmQuery() const
{
	if (!mQueryValid) {
		// record type remains at ANY
		mQuery.conjunctive(CSSM_DB_AND);
		for (vector<Comparison>::const_iterator it = mRelations.begin(); it != mRelations.end(); it++) {
			CssmSelectionPredicate pred;
			pred.dbOperator(it->mOperator);
			pred.attribute().info() = CssmDbAttributeInfo(it->mName.c_str(), it->mFormat);
			pred.attribute().set(it->mValue.get());
			mPredicates.push_back(pred);
		}
		mQuery.set((uint32)mPredicates.size(), &mPredicates[0]);
		mQueryValid = true;
	}
	return mQuery;
}


} // end namespace CssmClient
} // end namespace Security
