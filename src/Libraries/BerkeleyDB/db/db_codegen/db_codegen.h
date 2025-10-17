/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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

typedef struct __db_obj {
	char	 *name;			/* Database name */
	char	 *dbtype;		/* Database type */

	u_int32_t extentsize;		/* Queue: extent size */
	u_int32_t pagesize;		/* Pagesize */
	u_int32_t re_len;		/* Queue/Recno: record length */

	char	 *key_type;		/* Key type */
	int	  custom;		/* Custom key comparison. */

	char	 *primary;		/* Secondary: primary's name */
	u_int32_t secondary_len;	/* secondary: length */
	u_int32_t secondary_off;	/* secondary: 0-based byte offset */

	int	  dupsort;		/* Sorted duplicates */
	int	  recnum;		/* Btree: record numbers */
	int	  transaction;		/* Database is transactional */

	TAILQ_ENTRY(__db_obj) q;	/* List of databases */
} DB_OBJ;

typedef struct __env_obj {
	char	 *prefix;		/* Name prefix */
	char	 *home;			/* Environment home */

	u_int32_t gbytes;		/* GB, B of cache */
	u_int32_t bytes;
	u_int32_t ncache;		/* Number of caches */

	int	  private;		/* Private environment */
	int	  standalone;		/* Standalone database */
	int	  transaction;		/* Database is transactional */

	TAILQ_ENTRY(__env_obj) q;	/* List of environments, databases */
	TAILQ_HEAD(__head_db, __db_obj) dbq;
} ENV_OBJ;

TAILQ_HEAD(__head_env, __env_obj);	/* List of environments */
extern struct __head_env env_tree;

extern const char *progname;		/* Program name */

int api_c __P((char *));
#ifdef DEBUG
int parse_dump __P((void));
#endif
int parse_input __P((FILE *));
