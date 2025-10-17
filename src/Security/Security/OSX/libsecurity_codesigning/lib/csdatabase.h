/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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
// csdb - system-supported Code Signing related database interfaces
//
#ifndef _H_CSDATABASE
#define _H_CSDATABASE

#include "diskrep.h"
#include "sigblob.h"
#include <Security/Security.h>
#include <security_utilities/globalizer.h>
#include <security_utilities/sqlite++.h>
#include <security_utilities/cfutilities.h>


namespace Security {
namespace CodeSigning {

namespace SQLite = SQLite3;


class SignatureDatabase : public SQLite::Database {
public:
	SignatureDatabase(const char *path = defaultPath,
		int flags = SQLITE_OPEN_READONLY | SQLITE_OPEN_NOFOLLOW);
	virtual ~SignatureDatabase();
	
	FilterRep *findCode(DiskRep *rep);

public:
	static const char defaultPath[];
};


class SignatureDatabaseWriter : public SignatureDatabase {
public:
	SignatureDatabaseWriter(const char *path = defaultPath,
		int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOFOLLOW)
		: SignatureDatabase(path, flags) { }

	void storeCode(const BlobCore *sig, const char *location);
	
private:
	SQLite::int64 insertGlobal(const char *location, const BlobCore *blob);
	void insertCode(SQLite::int64 globid, int arch, const EmbeddedSignatureBlob *sig);
};


} // end namespace CodeSigning
} // end namespace Security

#endif // !_H_CSDATABASE
