/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 24, 2025.
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

#ifndef PARSE_INCLUDE
#define PARSE_INCLUDE

/* macros for the   parse_args   routine */

#define P_STRING 1		/* Macros for the result_type attribute */
#define P_CHAR 2
#define P_SHORT 3
#define P_INT 4
#define P_LONG 5
#define P_FILE 6
#define P_OLD_FILE 7
#define P_NEW_FILE 8
#define P_FLOAT 9
#define P_DOUBLE 10

#define P_CASE_INSENSITIVE 01	/* Macros for the   flags   attribute */
#define P_REQUIRED_PREFIX 02

#define P_NO_ARGS 0		/* Macros for the   arg_count   attribute */
#define P_ONE_ARG 1
#define P_INFINITE_ARGS 2

#define p_entry(pref,swit,flag,count,type,store,size) \
    { (pref), (swit), (flag), (count), (type), (int *) (store), (size) }

typedef struct {
    char *prefix;
    char *string;
    int flags;
    int count;
    int result_type;
    int *result_ptr;
    int table_size;
} arg_info;

#ifdef KR_headers
#define Argdcl(x) ()
#else
#define Argdcl(x) x
#endif
int	arg_verify Argdcl((char**, arg_info*, int));
void	init_store Argdcl((arg_info*, int));
int	match_table Argdcl((char*, arg_info*, int, int, int*));
int	parse_args Argdcl((int, char**, arg_info*, int, char**, int));

#endif
