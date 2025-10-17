/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 31, 2025.
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
#include "db_config.h"

#include "db_int.h"
#include "dbinc/crypto.h"
#include "dbinc/db_page.h"
#include "dbinc/db_am.h"
#include "dbinc/log.h"
#include "dbinc/txn.h"
#include "dbinc/fop.h"

/*
 * PUBLIC: int __fop_create_print __P((ENV *, DBT *, DB_LSN *,
 * PUBLIC:     db_recops, void *));
 */
int
__fop_create_print(env, dbtp, lsnp, notused2, notused3)
	ENV *env;
	DBT *dbtp;
	DB_LSN *lsnp;
	db_recops notused2;
	void *notused3;
{
	__fop_create_args *argp;
	u_int32_t i;
	int ch;
	int ret;

	notused2 = DB_TXN_PRINT;
	notused3 = NULL;

	if ((ret = __fop_create_read(env, dbtp->data, &argp)) != 0)
		return (ret);
	(void)printf(
    "[%lu][%lu]__fop_create%s: rec: %lu txnp %lx prevlsn [%lu][%lu]\n",
	    (u_long)lsnp->file, (u_long)lsnp->offset,
	    (argp->type & DB_debug_FLAG) ? "_debug" : "",
	    (u_long)argp->type,
	    (u_long)argp->txnp->txnid,
	    (u_long)argp->prev_lsn.file, (u_long)argp->prev_lsn.offset);
	(void)printf("\tname: ");
	for (i = 0; i < argp->name.size; i++) {
		ch = ((u_int8_t *)argp->name.data)[i];
		printf(isprint(ch) || ch == 0x0a ? "%c" : "%#x ", ch);
	}
	(void)printf("\n");
	(void)printf("\tappname: %lu\n", (u_long)argp->appname);
	(void)printf("\tmode: %o\n", argp->mode);
	(void)printf("\n");
	__os_free(env, argp);
	return (0);
}

/*
 * PUBLIC: int __fop_remove_print __P((ENV *, DBT *, DB_LSN *,
 * PUBLIC:     db_recops, void *));
 */
int
__fop_remove_print(env, dbtp, lsnp, notused2, notused3)
	ENV *env;
	DBT *dbtp;
	DB_LSN *lsnp;
	db_recops notused2;
	void *notused3;
{
	__fop_remove_args *argp;
	u_int32_t i;
	int ch;
	int ret;

	notused2 = DB_TXN_PRINT;
	notused3 = NULL;

	if ((ret = __fop_remove_read(env, dbtp->data, &argp)) != 0)
		return (ret);
	(void)printf(
    "[%lu][%lu]__fop_remove%s: rec: %lu txnp %lx prevlsn [%lu][%lu]\n",
	    (u_long)lsnp->file, (u_long)lsnp->offset,
	    (argp->type & DB_debug_FLAG) ? "_debug" : "",
	    (u_long)argp->type,
	    (u_long)argp->txnp->txnid,
	    (u_long)argp->prev_lsn.file, (u_long)argp->prev_lsn.offset);
	(void)printf("\tname: ");
	for (i = 0; i < argp->name.size; i++) {
		ch = ((u_int8_t *)argp->name.data)[i];
		printf(isprint(ch) || ch == 0x0a ? "%c" : "%#x ", ch);
	}
	(void)printf("\n");
	(void)printf("\tfid: ");
	for (i = 0; i < argp->fid.size; i++) {
		ch = ((u_int8_t *)argp->fid.data)[i];
		printf(isprint(ch) || ch == 0x0a ? "%c" : "%#x ", ch);
	}
	(void)printf("\n");
	(void)printf("\tappname: %lu\n", (u_long)argp->appname);
	(void)printf("\n");
	__os_free(env, argp);
	return (0);
}

/*
 * PUBLIC: int __fop_write_print __P((ENV *, DBT *, DB_LSN *,
 * PUBLIC:     db_recops, void *));
 */
int
__fop_write_print(env, dbtp, lsnp, notused2, notused3)
	ENV *env;
	DBT *dbtp;
	DB_LSN *lsnp;
	db_recops notused2;
	void *notused3;
{
	__fop_write_args *argp;
	u_int32_t i;
	int ch;
	int ret;

	notused2 = DB_TXN_PRINT;
	notused3 = NULL;

	if ((ret = __fop_write_read(env, dbtp->data, &argp)) != 0)
		return (ret);
	(void)printf(
    "[%lu][%lu]__fop_write%s: rec: %lu txnp %lx prevlsn [%lu][%lu]\n",
	    (u_long)lsnp->file, (u_long)lsnp->offset,
	    (argp->type & DB_debug_FLAG) ? "_debug" : "",
	    (u_long)argp->type,
	    (u_long)argp->txnp->txnid,
	    (u_long)argp->prev_lsn.file, (u_long)argp->prev_lsn.offset);
	(void)printf("\tname: ");
	for (i = 0; i < argp->name.size; i++) {
		ch = ((u_int8_t *)argp->name.data)[i];
		printf(isprint(ch) || ch == 0x0a ? "%c" : "%#x ", ch);
	}
	(void)printf("\n");
	(void)printf("\tappname: %lu\n", (u_long)argp->appname);
	(void)printf("\tpgsize: %lu\n", (u_long)argp->pgsize);
	(void)printf("\tpageno: %lu\n", (u_long)argp->pageno);
	(void)printf("\toffset: %lu\n", (u_long)argp->offset);
	(void)printf("\tpage: ");
	for (i = 0; i < argp->page.size; i++) {
		ch = ((u_int8_t *)argp->page.data)[i];
		printf(isprint(ch) || ch == 0x0a ? "%c" : "%#x ", ch);
	}
	(void)printf("\n");
	(void)printf("\tflag: %lu\n", (u_long)argp->flag);
	(void)printf("\n");
	__os_free(env, argp);
	return (0);
}

/*
 * PUBLIC: int __fop_rename_print __P((ENV *, DBT *, DB_LSN *,
 * PUBLIC:     db_recops, void *));
 */
int
__fop_rename_print(env, dbtp, lsnp, notused2, notused3)
	ENV *env;
	DBT *dbtp;
	DB_LSN *lsnp;
	db_recops notused2;
	void *notused3;
{
	__fop_rename_args *argp;
	u_int32_t i;
	int ch;
	int ret;

	notused2 = DB_TXN_PRINT;
	notused3 = NULL;

	if ((ret = __fop_rename_read(env, dbtp->data, &argp)) != 0)
		return (ret);
	(void)printf(
    "[%lu][%lu]__fop_rename%s: rec: %lu txnp %lx prevlsn [%lu][%lu]\n",
	    (u_long)lsnp->file, (u_long)lsnp->offset,
	    (argp->type & DB_debug_FLAG) ? "_debug" : "",
	    (u_long)argp->type,
	    (u_long)argp->txnp->txnid,
	    (u_long)argp->prev_lsn.file, (u_long)argp->prev_lsn.offset);
	(void)printf("\toldname: ");
	for (i = 0; i < argp->oldname.size; i++) {
		ch = ((u_int8_t *)argp->oldname.data)[i];
		printf(isprint(ch) || ch == 0x0a ? "%c" : "%#x ", ch);
	}
	(void)printf("\n");
	(void)printf("\tnewname: ");
	for (i = 0; i < argp->newname.size; i++) {
		ch = ((u_int8_t *)argp->newname.data)[i];
		printf(isprint(ch) || ch == 0x0a ? "%c" : "%#x ", ch);
	}
	(void)printf("\n");
	(void)printf("\tfileid: ");
	for (i = 0; i < argp->fileid.size; i++) {
		ch = ((u_int8_t *)argp->fileid.data)[i];
		printf(isprint(ch) || ch == 0x0a ? "%c" : "%#x ", ch);
	}
	(void)printf("\n");
	(void)printf("\tappname: %lu\n", (u_long)argp->appname);
	(void)printf("\n");
	__os_free(env, argp);
	return (0);
}

/*
 * PUBLIC: int __fop_file_remove_print __P((ENV *, DBT *, DB_LSN *,
 * PUBLIC:     db_recops, void *));
 */
int
__fop_file_remove_print(env, dbtp, lsnp, notused2, notused3)
	ENV *env;
	DBT *dbtp;
	DB_LSN *lsnp;
	db_recops notused2;
	void *notused3;
{
	__fop_file_remove_args *argp;
	u_int32_t i;
	int ch;
	int ret;

	notused2 = DB_TXN_PRINT;
	notused3 = NULL;

	if ((ret = __fop_file_remove_read(env, dbtp->data, &argp)) != 0)
		return (ret);
	(void)printf(
    "[%lu][%lu]__fop_file_remove%s: rec: %lu txnp %lx prevlsn [%lu][%lu]\n",
	    (u_long)lsnp->file, (u_long)lsnp->offset,
	    (argp->type & DB_debug_FLAG) ? "_debug" : "",
	    (u_long)argp->type,
	    (u_long)argp->txnp->txnid,
	    (u_long)argp->prev_lsn.file, (u_long)argp->prev_lsn.offset);
	(void)printf("\treal_fid: ");
	for (i = 0; i < argp->real_fid.size; i++) {
		ch = ((u_int8_t *)argp->real_fid.data)[i];
		printf(isprint(ch) || ch == 0x0a ? "%c" : "%#x ", ch);
	}
	(void)printf("\n");
	(void)printf("\ttmp_fid: ");
	for (i = 0; i < argp->tmp_fid.size; i++) {
		ch = ((u_int8_t *)argp->tmp_fid.data)[i];
		printf(isprint(ch) || ch == 0x0a ? "%c" : "%#x ", ch);
	}
	(void)printf("\n");
	(void)printf("\tname: ");
	for (i = 0; i < argp->name.size; i++) {
		ch = ((u_int8_t *)argp->name.data)[i];
		printf(isprint(ch) || ch == 0x0a ? "%c" : "%#x ", ch);
	}
	(void)printf("\n");
	(void)printf("\tappname: %lu\n", (u_long)argp->appname);
	(void)printf("\tchild: 0x%lx\n", (u_long)argp->child);
	(void)printf("\n");
	__os_free(env, argp);
	return (0);
}

/*
 * PUBLIC: int __fop_init_print __P((ENV *, DB_DISTAB *));
 */
int
__fop_init_print(env, dtabp)
	ENV *env;
	DB_DISTAB *dtabp;
{
	int ret;

	if ((ret = __db_add_recovery_int(env, dtabp,
	    __fop_create_print, DB___fop_create)) != 0)
		return (ret);
	if ((ret = __db_add_recovery_int(env, dtabp,
	    __fop_remove_print, DB___fop_remove)) != 0)
		return (ret);
	if ((ret = __db_add_recovery_int(env, dtabp,
	    __fop_write_print, DB___fop_write)) != 0)
		return (ret);
	if ((ret = __db_add_recovery_int(env, dtabp,
	    __fop_rename_print, DB___fop_rename)) != 0)
		return (ret);
	if ((ret = __db_add_recovery_int(env, dtabp,
	    __fop_rename_print, DB___fop_rename_noundo)) != 0)
		return (ret);
	if ((ret = __db_add_recovery_int(env, dtabp,
	    __fop_file_remove_print, DB___fop_file_remove)) != 0)
		return (ret);
	return (0);
}
