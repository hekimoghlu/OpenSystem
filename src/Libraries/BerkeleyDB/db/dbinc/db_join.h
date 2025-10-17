/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 26, 2024.
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
#ifndef _DB_JOIN_H_
#define	_DB_JOIN_H_

#if defined(__cplusplus)
extern "C" {
#endif

/*
 * Joins use a join cursor that is similar to a regular DB cursor except
 * that it only supports c_get and c_close functionality.  Also, it does
 * not support the full range of flags for get.
 */
typedef struct __join_cursor {
	u_int8_t *j_exhausted;	/* Array of flags; is cursor i exhausted? */
	DBC	**j_curslist;	/* Array of cursors in the join: constant. */
	DBC	**j_fdupcurs;	/* Cursors w/ first instances of current dup. */
	DBC	**j_workcurs;	/* Scratch cursor copies to muck with. */
	DB	 *j_primary;	/* Primary dbp. */
	DBT	  j_key;	/* Used to do lookups. */
	DBT	  j_rdata;	/* Memory used for data return. */
	u_int32_t j_ncurs;	/* How many cursors do we have? */
#define	JOIN_RETRY	0x01	/* Error on primary get; re-return same key. */
	u_int32_t flags;
} JOIN_CURSOR;

#if defined(__cplusplus)
}
#endif
#endif /* !_DB_JOIN_H_ */
