/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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
/*
   File:      mdspriv.h

   Contains:  Module Directory Services Data Types and API, private section.

*/

#ifndef _MDSPRIV_H_
#define _MDSPRIV_H_  1

#include <Security/mds.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
	const char *guid;
	uint32 ssid;
	const char *serial;
	const char *printName;
} MDS_InstallDefaults;


/* MDS Context APIs */

CSSM_RETURN CSSMAPI
MDS_InstallFile(MDS_HANDLE inMDSHandle, const MDS_InstallDefaults *defaults,
	const char *bundlePath, const char *subdir, const char *file);

CSSM_RETURN CSSMAPI
MDS_RemoveSubservice(MDS_HANDLE inMDSHandle, const char *guid, uint32 ssid);

#ifdef __cplusplus
}
#endif

#endif /* _MDS_H_ */
