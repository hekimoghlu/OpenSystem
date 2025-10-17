/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 26, 2022.
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
#ifndef ISC_MUTEX_H
#define ISC_MUTEX_H 1

#include <isc/result.h>		/* for ISC_R_ codes */

typedef int isc_mutex_t;

#define isc_mutex_init(mp) \
	(*(mp) = 0, ISC_R_SUCCESS)
#define isc_mutex_lock(mp) \
	((*(mp))++ == 0 ? ISC_R_SUCCESS : ISC_R_UNEXPECTED)
#define isc_mutex_unlock(mp) \
	(--(*(mp)) == 0 ? ISC_R_SUCCESS : ISC_R_UNEXPECTED)
#define isc_mutex_trylock(mp) \
	(*(mp) == 0 ? ((*(mp))++, ISC_R_SUCCESS) : ISC_R_LOCKBUSY)
#define isc_mutex_destroy(mp) \
	(*(mp) == 0 ? (*(mp) = -1, ISC_R_SUCCESS) : ISC_R_UNEXPECTED)
#define isc_mutex_stats(fp)

#endif /* ISC_MUTEX_H */
