/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 12, 2022.
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
/* $Id: snprintf-test.h 10377 2001-07-19 18:39:14Z assar $ */

#ifndef __STRFTIME_TEST_H__
#define __STRFTIME_TEST_H__

/*
 * we cannot use the real names of the functions when testing, since
 * they might have different prototypes as the system functions, hence
 * these evil hacks
 */

#define strftime test_strftime
#define strptime test_strptime

ROKEN_LIB_FUNCTION size_t ROKEN_LIB_CALL
strftime (char *buf, size_t maxsize, const char *format,
          const struct tm *tm);

ROKEN_LIB_FUNCTION char * ROKEN_LIB_CALL
strptime (const char *buf, const char *format, struct tm *timeptr);

#endif /* __STRFTIME_TEST_H__ */
