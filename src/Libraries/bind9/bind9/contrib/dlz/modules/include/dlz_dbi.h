/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 21, 2023.
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
#include <dlz_minimal.h>
#include <dlz_list.h>
#include <dlz_pthread.h>

#ifndef DLZ_DBI_H
#define DLZ_DBI_H 1

/*
 * Types
 */
#define REQUIRE_CLIENT	0x01
#define REQUIRE_QUERY	0x02
#define REQUIRE_RECORD	0x04
#define REQUIRE_ZONE	0x08

typedef struct query_segment query_segment_t;
typedef DLZ_LIST(query_segment_t) query_list_t;
typedef struct dbinstance dbinstance_t;
typedef DLZ_LIST(dbinstance_t) db_list_t;
typedef struct driverinstance driverinstance_t;

/*%
 * a query segment is all the text between our special tokens
 * special tokens are %zone%, %record%, %client%
 */
struct query_segment {
	void				*cmd;
	unsigned int			strlen;
	isc_boolean_t			direct;
	DLZ_LINK(query_segment_t)	link;
};

/*%
 * a database instance contains everything we need for running
 * a query against the database.  Using it each separate thread
 * can dynamically construct a query and execute it against the
 * database.  The "instance_lock" and locking code in the driver's
 * make sure no two threads try to use the same DBI at a time.
 */
struct dbinstance {
	void			*dbconn;
	query_list_t		*allnodes_q;
	query_list_t		*allowxfr_q;
	query_list_t		*authority_q;
	query_list_t		*findzone_q;
	query_list_t		*lookup_q;
	query_list_t		*countzone_q;
	char			*query_buf;
	char			*zone;
	char			*record;
	char			*client;
	dlz_mutex_t		lock;
	DLZ_LINK(dbinstance_t)	link;
};

/*
 * Method declarations
 */

void
destroy_querylist(query_list_t **querylist);

isc_result_t
build_querylist(const char *query_str, char **zone, char **record,
		char **client, query_list_t **querylist, unsigned int flags, 
		log_t log);

char *
build_querystring(query_list_t *querylist);

isc_result_t
build_dbinstance(const char *allnodes_str, const char *allowxfr_str,
		 const char *authority_str, const char *findzone_str,
		 const char *lookup_str, const char *countzone_str,
		 dbinstance_t **dbi, log_t log);

void
destroy_dbinstance(dbinstance_t *dbi);

char *
get_parameter_value(const char *input, const char* key);

#endif /* DLZ_DBI_H */
