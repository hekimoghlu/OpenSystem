/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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
	File:		macOnly.h

	Contains:	Mac-specific #defines for Yarrow.

	Written by:	Doug Mitchell

	Copyright: (c) 2000 by Apple Computer, Inc., all rights reserved.

	Change History (most recent first):

		02/10/99	dpm		Created.
 
*/

#if		!defined(macintosh) && !defined(__APPLE__)
#error Hey, why are you including macOnly for a non-Mac build!?
#endif

#ifndef	_MAC_ONLY_H_
#define _MAC_ONLY_H_

#include "WindowsTypesForMac.h"

#if defined(__cplusplus)
extern "C" {
#endif

/*
 * No "slow poll" for Mac. 
 */
#define SLOW_POLL_ENABLE	0
#if		SLOW_POLL_ENABLE
extern DWORD prng_slow_poll(BYTE* buf,UINT bufsize);
#endif	/* SLOW_POLL_ENABLE */

#if defined(__cplusplus)
}
#endif

#endif	/* _MAC_ONLY_H_*/
