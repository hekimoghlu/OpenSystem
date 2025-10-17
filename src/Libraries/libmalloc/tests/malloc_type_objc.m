/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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

#import <Foundation/Foundation.h>

#include <stdio.h>
#include <assert.h>

#include "tmo_test_defs.h"

void
validate_swift_obj_array(void **ptrs);

void
validate_obj_array(NSArray *a)
{
	void *ptrs[N_TEST_SWIFT_CLASSES];
	int i = 0;
	for (id obj in a) {
		ptrs[i] = (void *)obj;
		i++;
	}
	assert(i == N_TEST_SWIFT_CLASSES);

	validate_swift_obj_array(ptrs);
}
