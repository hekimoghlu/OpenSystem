/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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
// dbm++ - generic C++ layer interface to [n]dbm
//
#ifndef _H_DBMPP
#define _H_DBMPP

#include <security_utilities/utilities.h>
#include <security_cdsa_utilities/cssmdata.h>
#include <security_utilities/unix++.h>
#include <string>
#include <db.h>


namespace Security {
namespace UnixPlusPlus {


class UnixDb : public FileDesc {
public:
	UnixDb();
	UnixDb(const char *path, int flags = O_RDWR, int mode = 0666, DBTYPE type = DB_HASH);
	UnixDb(const std::string &path, int flags = O_RDWR, int mode = 0666, DBTYPE type = DB_HASH);
	
	virtual ~UnixDb();
		
	void open(const char *path, int flags = O_RDWR, int mode = 0666, DBTYPE type = DB_HASH);
	void open(const std::string &path, int flags = O_RDWR, int mode = 0666, DBTYPE type = DB_HASH);
	void close();

	bool get(const CssmData &key, CssmData &value, int flags = 0) const;
	bool get(const CssmData &key, CssmOwnedData &value, int flags = 0) const;
	bool put(const CssmData &key, const CssmData &value, int flags = 0);
	void erase(const CssmData &key, int flags = 0);
	void flush(int flags = 0);
	
	bool next(CssmData &key, CssmData &value, int flags = R_NEXT) const;
	bool first(CssmData &key, CssmData &value) const
		{ return next(key, value, R_FIRST); }
	
	operator bool () const
		{ return mDb; }
	
public:
	struct Data : public PodWrapper<Data, DBT> {
		template <class T>
		Data(const T &src)		{ DBT::data = src.data(); DBT::size = src.length(); }
		
		Data() { }
		Data(void *data, size_t length) { DBT::data = data; DBT::size = length; }
		Data(const DBT &dat)	{ DBT::data = dat.data; DBT::size = dat.size; }
		
		void *data() const		{ return DBT::data; }
		size_t length() const	{ return size; }
		operator bool () const	{ return DBT::data != NULL; }
		operator CssmData () const { return CssmData(data(), length()); }
	};

private:
	DB *mDb;
};


}	// end namespace UnixPlusPlus
}	// end namespace Security


#endif //_H_DBMPP
