/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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
#include "db++.h"
#include <security_utilities/debugging.h>


namespace Security {
namespace UnixPlusPlus {

UnixDb::UnixDb() : mDb(NULL)
{
}

UnixDb::UnixDb(const char *path, int flags, int mode, DBTYPE type) : mDb(NULL)
{
	open(path, flags, mode);
}

UnixDb::UnixDb(const std::string &path, int flags, int mode, DBTYPE type) : mDb(NULL)
{
	open(path, flags, mode);
}
	
UnixDb::~UnixDb()
{
	close();
}

void UnixDb::open(const char *path, int flags, int mode, DBTYPE type)
{
	if (DB* newDb = ::dbopen(path, flags, mode, type, NULL)) {
		close();
		mDb = newDb;
		setFd(mDb->fd(mDb));
		secnotice("unixdb", "open(%s,0x%x,0x%x,type=%d)=%p", path, flags, mode, type, mDb);
	} else
		UnixError::throwMe();
}

void UnixDb::open(const std::string &path, int flags, int mode, DBTYPE type)
{
	open(path.c_str(), flags, mode);
}

void UnixDb::close()
{
	if (mDb) {
		secnotice("unixdb", "close(%p)", mDb);
		mDb->close(mDb);
		mDb = NULL;
		setFd(invalidFd);
	}
}

bool UnixDb::get(const CssmData &key, CssmData &value, int flags) const
{
	Data dKey(key);
	Data val;
	int rc = mDb->get(mDb, &dKey, &val, flags);
	secnotice("unixdb", "get(%p,[:%ld],flags=0x%x)=%d[:%ld]",
		mDb, key.length(), flags, rc, value.length());
	checkError(rc);
	if (!rc) {
		value = val;
		return true;
	} else
		return false;
}

bool UnixDb::get(const CssmData &key, CssmOwnedData &value, int flags) const
{
	CssmData val;
	if (get(key, val, flags)) {
		value = val;
		return true;
	} else
		return false;
}

bool UnixDb::put(const CssmData &key, const CssmData &value, int flags)
{
	Data dKey(key);
	Data dValue(value);
	int rc = mDb->put(mDb, &dKey, &dValue, flags);
	secnotice("unixdb", "put(%p,[:%ld],[:%ld],flags=0x%x)=%d",
		mDb, key.length(), value.length(), flags, rc);
	checkError(rc);
	return !rc;
}

void UnixDb::erase(const CssmData &key, int flags)
{
	Data dKey(key);
	secnotice("unixdb", "delete(%p,[:%ld],flags=0x%x)", mDb, key.length(), flags);
	checkError(mDb->del(mDb, &dKey, flags));
}

bool UnixDb::next(CssmData &key, CssmData &value, int flags /* = R_NEXT */) const
{
	Data dKey, dValue;
	int rc = mDb->seq(mDb, &dKey, &dValue, flags);
	checkError(rc);
	if (!rc) {
		key = dKey;
		value = dValue;
		return true;
	} else
		return false;
}


void UnixDb::flush(int flags)
{
	checkError(mDb->sync(mDb, flags));
}


}	// end namespace UnixPlusPlus
}	// end namespace Security
