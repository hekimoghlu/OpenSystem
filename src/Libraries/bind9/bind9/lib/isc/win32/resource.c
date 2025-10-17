/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 23, 2024.
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
/* $Id: resource.c,v 1.10 2008/07/11 23:47:09 tbox Exp $ */

#include <config.h>

#include <stdio.h>

#include <isc/platform.h>
#include <isc/resource.h>
#include <isc/result.h>
#include <isc/util.h>

#include "errno2result.h"

/*
 * Windows limits the maximum number of open files to 2048
 */

#define WIN32_MAX_OPEN_FILES	2048

isc_result_t
isc_resource_setlimit(isc_resource_t resource, isc_resourcevalue_t value) {
	isc_resourcevalue_t rlim_value;
	int wresult;

	if (resource != isc_resource_openfiles)
		return (ISC_R_NOTIMPLEMENTED);


	if (value == ISC_RESOURCE_UNLIMITED)
		rlim_value = WIN32_MAX_OPEN_FILES;
	else
		rlim_value = min(value, WIN32_MAX_OPEN_FILES);

	wresult = _setmaxstdio((int) rlim_value);

	if (wresult > 0)
		return (ISC_R_SUCCESS);
	else
		return (isc__errno2result(errno));
}

isc_result_t
isc_resource_getlimit(isc_resource_t resource, isc_resourcevalue_t *value) {

	if (resource != isc_resource_openfiles)
		return (ISC_R_NOTIMPLEMENTED);

	*value = WIN32_MAX_OPEN_FILES;
	return (ISC_R_SUCCESS);
}

isc_result_t
isc_resource_getcurlimit(isc_resource_t resource, isc_resourcevalue_t *value) {
	return (isc_resource_getlimit(resource, value));
}
