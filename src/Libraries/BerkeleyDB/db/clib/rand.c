/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 16, 2025.
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

/*
 * rand, srand --
 *
 * PUBLIC: #ifndef HAVE_RAND
 * PUBLIC: int rand __P((void));
 * PUBLIC: void srand __P((unsigned int));
 * PUBLIC: #endif
 */
int rand(void)	/* RAND_MAX assumed to be 32767 */
{
	DB_GLOBAL(rand_next) = DB_GLOBAL(rand_next) * 1103515245 + 12345;
	return (unsigned int) (DB_GLOBAL(rand_next)/65536) % 32768;
}

void srand(unsigned int seed)
{
	DB_GLOBAL(rand_next) = seed;
}
