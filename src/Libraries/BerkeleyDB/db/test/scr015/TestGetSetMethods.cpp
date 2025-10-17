/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 5, 2025.
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
/*
 * Do some regression tests for simple get/set access methods
 * on DbEnv, DbTxn, Db.  We don't currently test that they have
 * the desired effect, only that they operate and return correctly.
 */

#include <db_cxx.h>
#include <iostream.h>

int main(int argc, char *argv[])
{
	try {
		DbEnv *dbenv = new DbEnv(0);
		DbTxn *dbtxn;
		u_int8_t conflicts[10];

		dbenv->set_error_stream(&cerr);
		dbenv->set_timeout(0x90000000,
				   DB_SET_LOCK_TIMEOUT);
		dbenv->set_lg_bsize(0x1000);
		dbenv->set_lg_dir(".");
		dbenv->set_lg_max(0x10000000);
		dbenv->set_lg_regionmax(0x100000);
		dbenv->set_lk_conflicts(conflicts, sizeof(conflicts));
		dbenv->set_lk_detect(DB_LOCK_DEFAULT);
		dbenv->set_lk_max_lockers(100);
		dbenv->set_lk_max_locks(10);
		dbenv->set_lk_max_objects(1000);
		dbenv->set_mp_mmapsize(0x10000);

		// Need to open the environment so we
		// can get a transaction.
		//
		dbenv->open(".", DB_CREATE | DB_INIT_TXN |
			    DB_INIT_LOCK | DB_INIT_LOG |
			    DB_INIT_MPOOL,
			    0644);

		dbenv->txn_begin(NULL, &dbtxn, DB_TXN_NOWAIT);
		dbtxn->set_timeout(0xA0000000, DB_SET_TXN_TIMEOUT);
		dbtxn->abort();

		dbenv->close(0);

		// We get a db, one for each type.
		// That's because once we call (for instance)
		// set_bt_minkey, DB 'knows' that this is a
		// Btree Db, and it cannot be used to try Hash
		// or Recno functions.
		//
		Db *db_bt = new Db(NULL, 0);
		db_bt->set_bt_minkey(100);
		db_bt->set_cachesize(0, 0x100000, 0);
		db_bt->close(0);

		Db *db_h = new Db(NULL, 0);
		db_h->set_h_ffactor(0x10);
		db_h->set_h_nelem(100);
		db_h->set_lorder(0);
		db_h->set_pagesize(0x10000);
		db_h->close(0);

		Db *db_re = new Db(NULL, 0);
		db_re->set_re_delim('@');
		db_re->set_re_pad(10);
		db_re->set_re_source("re.in");
		db_re->close(0);

		Db *db_q = new Db(NULL, 0);
		db_q->set_q_extentsize(200);
		db_q->close(0);

	}
	catch (DbException &dbe) {
		cerr << "Db Exception: " << dbe.what() << "\n";
	}
	return 0;
}
