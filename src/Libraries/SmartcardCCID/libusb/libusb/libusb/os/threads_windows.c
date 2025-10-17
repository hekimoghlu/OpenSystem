/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 12, 2022.
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
#include "libusbi.h"

int usbi_cond_timedwait(usbi_cond_t *cond,
	usbi_mutex_t *mutex, const struct timeval *tv)
{
	DWORD millis;

	millis = (DWORD)(tv->tv_sec * 1000L) + (tv->tv_usec / 1000L);
	/* round up to next millisecond */
	if (tv->tv_usec % 1000L)
		millis++;

	if (SleepConditionVariableCS(cond, mutex, millis))
		return 0;
	else if (GetLastError() == ERROR_TIMEOUT)
		return LIBUSB_ERROR_TIMEOUT;
	else
		return LIBUSB_ERROR_OTHER;
}
