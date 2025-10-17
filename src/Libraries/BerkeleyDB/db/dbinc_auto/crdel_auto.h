/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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
#ifndef	__crdel_AUTO_H
#define	__crdel_AUTO_H
#define	DB___crdel_metasub	142
typedef struct ___crdel_metasub_args {
	u_int32_t type;
	DB_TXN *txnp;
	DB_LSN prev_lsn;
	int32_t	fileid;
	db_pgno_t	pgno;
	DBT	page;
	DB_LSN	lsn;
} __crdel_metasub_args;

#define	DB___crdel_inmem_create	138
typedef struct ___crdel_inmem_create_args {
	u_int32_t type;
	DB_TXN *txnp;
	DB_LSN prev_lsn;
	int32_t	fileid;
	DBT	name;
	DBT	fid;
	u_int32_t	pgsize;
} __crdel_inmem_create_args;

#define	DB___crdel_inmem_rename	139
typedef struct ___crdel_inmem_rename_args {
	u_int32_t type;
	DB_TXN *txnp;
	DB_LSN prev_lsn;
	DBT	oldname;
	DBT	newname;
	DBT	fid;
} __crdel_inmem_rename_args;

#define	DB___crdel_inmem_remove	140
typedef struct ___crdel_inmem_remove_args {
	u_int32_t type;
	DB_TXN *txnp;
	DB_LSN prev_lsn;
	DBT	name;
	DBT	fid;
} __crdel_inmem_remove_args;

#endif
