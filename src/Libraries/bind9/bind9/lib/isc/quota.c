/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 28, 2022.
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
/* $Id: quota.c,v 1.18 2007/06/19 23:47:17 tbox Exp $ */

/*! \file */

#include <config.h>

#include <stddef.h>

#include <isc/quota.h>
#include <isc/util.h>

isc_result_t
isc_quota_init(isc_quota_t *quota, int max) {
	quota->max = max;
	quota->used = 0;
	quota->soft = 0;
	return (isc_mutex_init(&quota->lock));
}

void
isc_quota_destroy(isc_quota_t *quota) {
	INSIST(quota->used == 0);
	quota->max = 0;
	quota->used = 0;
	quota->soft = 0;
	DESTROYLOCK(&quota->lock);
}

void
isc_quota_soft(isc_quota_t *quota, int soft) {
	LOCK(&quota->lock);
	quota->soft = soft;
	UNLOCK(&quota->lock);
}

void
isc_quota_max(isc_quota_t *quota, int max) {
	LOCK(&quota->lock);
	quota->max = max;
	UNLOCK(&quota->lock);
}

isc_result_t
isc_quota_reserve(isc_quota_t *quota) {
	isc_result_t result;
	LOCK(&quota->lock);
	if (quota->max == 0 || quota->used < quota->max) {
		if (quota->soft == 0 || quota->used < quota->soft)
			result = ISC_R_SUCCESS;
		else
			result = ISC_R_SOFTQUOTA;
		quota->used++;
	} else
		result = ISC_R_QUOTA;
	UNLOCK(&quota->lock);
	return (result);
}

void
isc_quota_release(isc_quota_t *quota) {
	LOCK(&quota->lock);
	INSIST(quota->used > 0);
	quota->used--;
	UNLOCK(&quota->lock);
}

isc_result_t
isc_quota_attach(isc_quota_t *quota, isc_quota_t **p)
{
	isc_result_t result;
	INSIST(p != NULL && *p == NULL);
	result = isc_quota_reserve(quota);
	if (result == ISC_R_SUCCESS || result == ISC_R_SOFTQUOTA)
		*p = quota;
	return (result);
}

void
isc_quota_detach(isc_quota_t **p)
{
	INSIST(p != NULL && *p != NULL);
	isc_quota_release(*p);
	*p = NULL;
}
