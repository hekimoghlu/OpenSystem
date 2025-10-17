/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 10, 2023.
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
#ifndef ISC_TM_H
#define ISC_TM_H 1

/*! \file isc/tm.h
 * Provides portable conversion routines for struct tm.
 */
#include <time.h>

#include <isc/lang.h>
#include <isc/types.h>


ISC_LANG_BEGINDECLS

time_t
isc_tm_timegm(struct tm *tm);
/*
 * Convert a tm structure to time_t, using UTC rather than the local
 * time zone.
 */

char *
isc_tm_strptime(const char *buf, const char *fmt, struct tm *tm);
/*
 * Parse a formatted date string into struct tm.
 */

ISC_LANG_ENDDECLS

#endif /* ISC_TIMER_H */
