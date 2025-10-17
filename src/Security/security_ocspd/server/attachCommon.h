/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 21, 2024.
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
 * attachCommon.h - attach/detach to/from arbitrary module
 */
 
#ifndef	_ATTACH_COMMON_H_
#define _ATTACH_COMMON_H_

#include <Security/cssmtype.h>

#ifdef __cplusplus
extern "C" {
#endif

/* load & attach; returns 0 on error */
CSSM_HANDLE attachCommon(
	const CSSM_GUID *guid,
	uint32 subserviceFlags);		// CSSM_SERVICE_TP, etc.

/* detach & unload */
void detachCommon(
	const CSSM_GUID *guid,
	CSSM_HANDLE hand);

#ifdef __cplusplus
}
#endif

#endif	/* _ATTACH_COMMON_H_ */

