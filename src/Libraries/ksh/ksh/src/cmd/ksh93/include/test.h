/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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
#pragma prototyped
#ifndef TEST_ARITH
/*
 *	UNIX shell
 *	David Korn
 *	AT&T Labs
 *
 */

#include	"FEATURE/options"
#include	"defs.h"
#include	"shtable.h"
/*
 *  These are the valid test operators
 */

#define TEST_ARITH	040	/* arithmetic operators */
#define TEST_BINOP	0200	/* binary operator */
#define TEST_PATTERN	0100	/* turn off bit for pattern compares */

#define TEST_NE		(TEST_ARITH|9)
#define TEST_EQ		(TEST_ARITH|4)
#define TEST_GE		(TEST_ARITH|5)
#define TEST_GT		(TEST_ARITH|6)
#define TEST_LE		(TEST_ARITH|7)
#define TEST_LT		(TEST_ARITH|8)
#define TEST_OR		(TEST_BINOP|1)
#define TEST_AND	(TEST_BINOP|2)
#define TEST_SNE	(TEST_PATTERN|1)
#define TEST_SEQ	(TEST_PATTERN|14)
#define TEST_PNE	1
#define TEST_PEQ	14
#define TEST_EF		3
#define TEST_NT		10
#define TEST_OT		12
#define TEST_SLT	16
#define TEST_SGT	17
#define TEST_END	8
#define TEST_REP	20

extern int test_unop(Shell_t*,int, const char*);
extern int test_inode(const char*, const char*);
extern int test_binop(Shell_t*,int, const char*, const char*);

extern const char	sh_opttest[];
extern const char	test_opchars[];
extern const char	e_argument[];
extern const char	e_missing[];
extern const char	e_badop[];
extern const char	e_tstbegin[];
extern const char	e_tstend[];

#endif /* TEST_ARITH */
