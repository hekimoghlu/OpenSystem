/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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
#include <stdlib.h>
#include <string.h>

#include <security/pam_appl.h>

#include "openpam_impl.h"

/*
 * OpenPAM extension
 *
 * Generic cleanup function
 */

void
openpam_free_data(pam_handle_t *pamh,
	void *data,
	int status)
{

	ENTER();
	(void)pamh;
	(void)status;
	FREE(data);
	RETURNV();
}

/*
 * Error codes:
 */

/**
 * The =openpam_free_data function is a cleanup function suitable for
 * passing to =pam_set_data.
 * It simply releases the data by passing its =data argument to =free.
 */
