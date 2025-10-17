/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 2, 2024.
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
extern DbRecord DbRecord_base;			/* Initialized structure. */

/*
 * Prototypes
 */
extern int	DbRecord_discard(DbRecord *);
extern int	DbRecord_init(const DBT *, const DBT *, DbRecord *);
extern void	DbRecord_print(DbRecord *, FILE *);
extern int	DbRecord_read(u_long, DbRecord *);
extern int	DbRecord_search_field_name(char *, char *, OPERATOR);
extern int	DbRecord_search_field_number(u_int, char *, OPERATOR);
extern int	compare_double(DB *, const DBT *, const DBT *);
extern int	compare_string(DB *, const DBT *, const DBT *);
extern int	compare_ulong(DB *, const DBT *, const DBT *);
extern int	csv_env_close(void);
extern int	csv_env_open(const char *, int);
extern int	csv_secondary_close(void);
extern int	csv_secondary_open(void);
extern int	entry_print(void *, size_t, u_int32_t);
extern int	field_cmp_double(void *, void *, OPERATOR);
extern int	field_cmp_re(void *, void *, OPERATOR);
extern int	field_cmp_string(void *, void *, OPERATOR);
extern int	field_cmp_ulong(void *, void *, OPERATOR);
extern int	input_load(input_fmt, u_long);
extern int	query(char *, int *);
extern int	query_interactive(void);
extern int	secondary_callback(DB *, const DBT *, const DBT *, DBT *);
extern int	strtod_err(char *, double *);
extern int	strtoul_err(char *, u_long *);
