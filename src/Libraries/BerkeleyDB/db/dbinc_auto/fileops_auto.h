/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 19, 2025.
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
#ifndef	__fop_AUTO_H
#define	__fop_AUTO_H
#define	DB___fop_create	143
typedef struct ___fop_create_args {
	u_int32_t type;
	DB_TXN *txnp;
	DB_LSN prev_lsn;
	DBT	name;
	u_int32_t	appname;
	u_int32_t	mode;
} __fop_create_args;

#define	DB___fop_remove	144
typedef struct ___fop_remove_args {
	u_int32_t type;
	DB_TXN *txnp;
	DB_LSN prev_lsn;
	DBT	name;
	DBT	fid;
	u_int32_t	appname;
} __fop_remove_args;

#define	DB___fop_write	145
typedef struct ___fop_write_args {
	u_int32_t type;
	DB_TXN *txnp;
	DB_LSN prev_lsn;
	DBT	name;
	u_int32_t	appname;
	u_int32_t	pgsize;
	db_pgno_t	pageno;
	u_int32_t	offset;
	DBT	page;
	u_int32_t	flag;
} __fop_write_args;

#define	DB___fop_rename	146
#define	DB___fop_rename_noundo	150
typedef struct ___fop_rename_args {
	u_int32_t type;
	DB_TXN *txnp;
	DB_LSN prev_lsn;
	DBT	oldname;
	DBT	newname;
	DBT	fileid;
	u_int32_t	appname;
} __fop_rename_args;

#define	DB___fop_file_remove	141
typedef struct ___fop_file_remove_args {
	u_int32_t type;
	DB_TXN *txnp;
	DB_LSN prev_lsn;
	DBT	real_fid;
	DBT	tmp_fid;
	DBT	name;
	u_int32_t	appname;
	u_int32_t	child;
} __fop_file_remove_args;

#endif
