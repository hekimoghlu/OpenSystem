/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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
/* $Id: once.c,v 1.12 2007/06/18 23:47:49 tbox Exp $ */

/* Principal Authors: DCL */

#include <config.h>

#include <windows.h>

#include <isc/once.h>
#include <isc/assertions.h>
#include <isc/util.h>

isc_result_t
isc_once_do(isc_once_t *controller, void(*function)(void)) {
	REQUIRE(controller != NULL && function != NULL);

	if (controller->status == ISC_ONCE_INIT_NEEDED) {

		if (InterlockedDecrement(&controller->counter) == 0) {
			if (controller->status == ISC_ONCE_INIT_NEEDED) {
				function();
				controller->status = ISC_ONCE_INIT_DONE;
			}
		} else {
			while (controller->status == ISC_ONCE_INIT_NEEDED) {
				/*
				 * Sleep(0) indicates that this thread 
				 * should be suspended to allow other 
				 * waiting threads to execute.
				 */
				Sleep(0);
			}
		}
	}

	return (ISC_R_SUCCESS);
}
