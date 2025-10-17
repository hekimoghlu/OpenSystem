/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 8, 2024.
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

/*! \file */

#include <config.h>

#include <isc/mutexblock.h>
#include <isc/util.h>

isc_result_t
isc_mutexblock_init(isc_mutex_t *block, unsigned int count) {
	isc_result_t result;
	unsigned int i;

	for (i = 0; i < count; i++) {
		result = isc_mutex_init(&block[i]);
		if (result != ISC_R_SUCCESS) {
			while (i > 0U) {
				i--;
				DESTROYLOCK(&block[i]);
			}
			return (result);
		}
	}

	return (ISC_R_SUCCESS);
}

isc_result_t
isc_mutexblock_destroy(isc_mutex_t *block, unsigned int count) {
	isc_result_t result;
	unsigned int i;

	for (i = 0; i < count; i++) {
		result = isc_mutex_destroy(&block[i]);
		if (result != ISC_R_SUCCESS)
			return (result);
	}

	return (ISC_R_SUCCESS);
}
