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
/*
 * Regress test for misc helper functions.
 *
 * Placed in the public domain.
 */

#include "includes.h"

#include <sys/types.h>
#include <stdio.h>
#ifdef HAVE_STDINT_H
#include <stdint.h>
#endif
#include <stdlib.h>
#include <string.h>

#include "../test_helper/test_helper.h"

#include "log.h"
#include "misc.h"

void test_parse(void);
void test_convtime(void);
void test_expand(void);
void test_argv(void);
void test_strdelim(void);
void test_hpdelim(void);
void test_ptimeout(void);

void
tests(void)
{
	test_parse();
	test_convtime();
	test_expand();
	test_argv();
	test_strdelim();
	test_hpdelim();
	test_ptimeout();
}
