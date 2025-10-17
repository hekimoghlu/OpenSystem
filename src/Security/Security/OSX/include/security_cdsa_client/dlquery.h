/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 30, 2021.
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

#ifndef _H_CDSA_CLIENT_DLQUERY
#define _H_CDSA_CLIENT_DLQUERY

#include <security_cdsa_utilities/cssmdb.h>
#include <string>
#include <vector>


namespace Security {
namespace CssmClient {


//
// A DL record attribute
//
class Attribute {
public:
	Attribute(const std::string &name) : mName(name) { }
	Attribute(const char *name) : mName(name) { }
	
	const std::string &name() const { return mName; }

private:
	std::string mName;
};


//
// A comparison (attribute ~rel~ constant-value)
//
class Comparison {
	friend class Query;
public:
	Comparison(const Attribute &attr, CSSM_DB_OPERATOR op, const char *s);
	Comparison(const Attribute &attr, CSSM_DB_OPERATOR op, const std::string &s);
	Comparison(const Attribute &attr, CSSM_DB_OPERATOR op, uint32 v);
	Comparison(const Attribute &attr, CSSM_DB_OPERATOR op, bool v);
	Comparison(const Attribute &attr, CSSM_DB_OPERATOR op, const CSSM_GUID &guid);
	Comparison(const Attribute &attr, CSSM_DB_OPERATOR op, const CssmData &data);
	
	Comparison(const Attribute &attr);
	friend Comparison operator ! (const Attribute &attr);
	
	Comparison(const Comparison &r);
	Comparison &operator = (const Comparison &r);
	
private:
	std::string mName;
	CSSM_DB_OPERATOR mOperator;
	CSSM_DB_ATTRIBUTE_FORMAT mFormat;
	CssmAutoData mValue;
};

template <class Value>
Comparison operator == (const Attribute &attr, const Value &value)
{ return Comparison(attr, CSSM_DB_EQUAL, value); }

template <class Value>
Comparison operator != (const Attribute &attr, const Value &value)
{ return Comparison(attr, CSSM_DB_NOT_EQUAL, value); }

template <class Value>
Comparison operator < (const Attribute &attr, const Value &value)
{ return Comparison(attr, CSSM_DB_LESS_THAN, value); }

template <class Value>
Comparison operator > (const Attribute &attr, const Value &value)
{ return Comparison(attr, CSSM_DB_GREATER_THAN, value); }

template <class Value>
Comparison operator % (const Attribute &attr, const Value &value)
{ return Comparison(attr, CSSM_DB_CONTAINS, value); }


//
// A Query
//
class Query {
public:
	Query() : mQueryValid(false) { }
	Query(const Comparison r) : mQueryValid(false) { mRelations.push_back(r); }
	Query(const Attribute &attr) : mQueryValid(false) { mRelations.push_back(attr); }
	
	Query(const Query &q) : mRelations(q.mRelations), mQueryValid(false) { }
	
	Query &operator = (const Query &q);
	
	Query &add(const Comparison &r)
	{ mRelations.push_back(r); return *this; }
	
	const CssmQuery &cssmQuery() const;

private:
	std::vector<Comparison> mRelations;
	
	// cached CssmQuery equivalent of this object
	mutable bool mQueryValid;   // mQuery has been constructed
	mutable vector<CssmSelectionPredicate> mPredicates; // holds lifetimes for mQuery
	mutable CssmQuery mQuery;
};

inline Query operator && (Query c, const Comparison &r)
{ return c.add(r); }


} // end namespace CssmClient
} // end namespace Security

#endif // _H_CDSA_CLIENT_DLQUERY
