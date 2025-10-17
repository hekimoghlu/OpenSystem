/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 24, 2023.
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

#include <db_cxx.h>

/*
 * Resource-acquisition-as-initialization pattern for Berkeley DB's cursors.
 *
 * Use DbcAuto instead of Berkeley DB's builtin Dbc class.  The constructor
 * allocates a new cursor, and it is freed automatically when it goes out of
 * scope.
 *
 * Note that some care is required with the order in which Berkeley DB handles
 * are closed.  In particular, the cursor handle must be closed before any
 * database or transaction handles the cursor references.  In addition, the
 * cursor close method can throw exceptions, which are masked by the destructor.
 * 
 * For these reasons, you are strongly advised to call the DbcAuto::close
 * method in the non-exceptional case.  This class exists to ensure that
 * cursors are closed if an exception occurs.
 */
class DbcAuto {
public:
	DbcAuto(Db *db, DbTxn *txn, u_int32_t flags) {
		db->cursor(txn, &dbc_, flags);
	}

	~DbcAuto() {
		try {
			close();
		} catch(...) {
			// Ignore it, another exception is pending
		}
	}

	void close() {
		if (dbc_) {
			// Set the member to 0 before making the call in
			// case an exception is thrown.
			Dbc *tdbc = dbc_;
			dbc_ = 0;
			tdbc->close();
		}
	}

	operator Dbc *() {
		return dbc_;
	}

	operator Dbc **() {
		return &dbc_;
	}

	Dbc *operator->() {
		return dbc_;
	}

private:
	Dbc *dbc_;
};
