/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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
// multidldb interfaces for searching multiple dls or db with a single cursor.
//
#ifndef _H_CDSA_CLIENT_MULTIDLDB
#define _H_CDSA_CLIENT_MULTIDLDB  1

#include <security_cdsa_client/dlclient.h>
#include <security_cdsa_client/DLDBList.h>

namespace Security
{

namespace CssmClient
{

//
// The MultiDLDb class.
//
class MultiDLDbImpl : public ObjectImpl, public DbCursorMaker
{
public:
	struct List : public vector<DLDbIdentifier>, public RefCount
	{
		List(const vector<DLDbIdentifier> &list) : vector<DLDbIdentifier>(list) {}
	};

	struct ListRef : public RefPointer<List>
	{
		ListRef() {}
		ListRef(const vector<DLDbIdentifier> &list) : RefPointer<List>(new List(list)) {}
	};

	MultiDLDbImpl(const vector<DLDbIdentifier> &list, bool useSecureStorage, const Cssm &cssm);
	MultiDLDbImpl(const vector<DLDbIdentifier> &list, bool useSecureStorage);
	virtual ~MultiDLDbImpl();

	Cssm cssm() const { return parent<Cssm>(); }
	Db database(const DLDbIdentifier &dlDbIdentifier);
	ListRef listRef() { return mListRef; }
	void list(const vector<DLDbIdentifier> &list);
    const vector<DLDbIdentifier> &list() { return *mListRef; }

	// DbCursorMaker
	virtual DbCursorImpl *newDbCursor(const CSSM_QUERY &query, Allocator &allocator);
	virtual DbCursorImpl *newDbCursor(uint32 capacity, Allocator &allocator);

protected:
	void activate();
	void deactivate();

private:
	typedef map<DLDbIdentifier, Db> DbMap;

	// Lock protecting this object during changes.
	Mutex mLock;
	ListRef mListRef;
	DbMap mDbMap;
	bool mUseSecureStorage;
};

class MultiDLDb : public Object
{
public:
	typedef MultiDLDbImpl Impl;

	explicit MultiDLDb(Impl *impl) : Object(impl) {}
	MultiDLDb(const vector<DLDbIdentifier> &list, bool useSecureStorage, const Cssm &cssm)
	: Object(new Impl(list, useSecureStorage, cssm)) {}
	MultiDLDb(const vector<DLDbIdentifier> &list, bool useSecureStorage)
	: Object(new Impl(list, useSecureStorage)) {}

	Impl *operator ->() const { return &impl<Impl>(); }
	Impl &operator *() const { return impl<Impl>(); }
	
	// Conversion to DbCursorMaker
	operator DbCursorMaker &() { return impl<Impl>(); }
};

}; // end namespace CssmClient

} // end namespace Security

#endif // _H_CDSA_CLIENT_MULTIDLDB
