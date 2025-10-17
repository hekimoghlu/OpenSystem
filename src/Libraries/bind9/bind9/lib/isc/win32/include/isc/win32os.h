/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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
/* $Id: win32os.h,v 1.7 2009/06/25 23:48:02 tbox Exp $ */

#ifndef ISC_WIN32OS_H
#define ISC_WIN32OS_H 1

#include <isc/lang.h>

ISC_LANG_BEGINDECLS

/*
 * Return the number of CPUs available on the system, or 1 if this cannot
 * be determined.
 */

int
isc_win32os_versioncheck(unsigned int major, unsigned int minor,
		     unsigned int updatemajor, unsigned int updateminor);

/*
 * Checks the current version of the operating system with the
 * supplied version information.
 * Returns:
 * -1	if less than the version information supplied
 *  0   if equal to all of the version information supplied
 * +1   if greater than the version information supplied
 */

ISC_LANG_ENDDECLS

#endif /* ISC_WIN32OS_H */
