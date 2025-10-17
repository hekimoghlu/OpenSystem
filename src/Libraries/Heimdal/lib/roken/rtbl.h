/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 23, 2025.
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
/* $Id$ */

#ifndef __rtbl_h__
#define __rtbl_h__

#ifndef ROKEN_LIB_FUNCTION
#ifdef _WIN32
#define ROKEN_LIB_FUNCTION
#define ROKEN_LIB_CALL     __cdecl
#else
#define ROKEN_LIB_FUNCTION
#define ROKEN_LIB_CALL
#endif
#endif

#if !defined(__GNUC__) && !defined(__attribute__)
#define __attribute__(x)
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct rtbl_data;
typedef struct rtbl_data *rtbl_t;

#define RTBL_ALIGN_LEFT		0
#define RTBL_ALIGN_RIGHT	1

/* flags */
#define RTBL_HEADER_STYLE_NONE	1
#define RTBL_JSON		2

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
rtbl_add_column (rtbl_t, const char*, unsigned int);

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
rtbl_add_column_by_id (rtbl_t, unsigned int, const char*, unsigned int);

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
rtbl_add_column_entryv_by_id (rtbl_t table, unsigned int id,
			      const char *fmt, ...)
	__attribute__ ((format (printf, 3, 0)));

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
rtbl_add_column_entry (rtbl_t, const char*, const char*);

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
rtbl_add_column_entryv (rtbl_t, const char*, const char*, ...)
	__attribute__ ((format (printf, 3, 0)));

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
rtbl_add_column_entry_by_id (rtbl_t, unsigned int, const char*);

ROKEN_LIB_FUNCTION rtbl_t ROKEN_LIB_CALL
rtbl_create (void);

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
rtbl_destroy (rtbl_t);

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
rtbl_format (rtbl_t, FILE*);

ROKEN_LIB_FUNCTION char * ROKEN_LIB_CALL
rtbl_format_str (rtbl_t);

ROKEN_LIB_FUNCTION unsigned int ROKEN_LIB_CALL
rtbl_get_flags (rtbl_t);

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
rtbl_new_row (rtbl_t);

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
rtbl_set_column_affix_by_id (rtbl_t, unsigned int, const char*, const char*);

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
rtbl_set_column_prefix (rtbl_t, const char*, const char*);

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
rtbl_set_flags (rtbl_t, unsigned int);

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
rtbl_set_prefix (rtbl_t, const char*);

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
rtbl_set_separator (rtbl_t, const char*);

#ifdef __cplusplus
}
#endif

#endif /* __rtbl_h__ */
