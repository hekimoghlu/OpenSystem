/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 24, 2024.
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
/* $Id: once.h,v 1.13 2007/06/19 23:47:18 tbox Exp $ */

#ifndef ISC_ONCE_H
#define ISC_ONCE_H 1

/*! \file */

#include <pthread.h>

#include <isc/platform.h>
#include <isc/result.h>

typedef pthread_once_t isc_once_t;

#ifdef ISC_PLATFORM_BRACEPTHREADONCEINIT
/*!
 * This accomodates systems that define PTHRAD_ONCE_INIT improperly.
 */
#define ISC_ONCE_INIT { PTHREAD_ONCE_INIT }
#else
/*!
 * This is the usual case.
 */
#define ISC_ONCE_INIT PTHREAD_ONCE_INIT
#endif

/* XXX We could do fancier error handling... */

#define isc_once_do(op, f) \
	((pthread_once((op), (f)) == 0) ? \
	 ISC_R_SUCCESS : ISC_R_UNEXPECTED)

#endif /* ISC_ONCE_H */
