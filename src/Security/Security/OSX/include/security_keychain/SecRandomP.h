/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 24, 2024.
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
/*!
	@header SecRandomP
    Provides an additional CFDataRef returning random function
 */

#ifndef _SECURITY_SECRANDOMP_H_
#define _SECURITY_SECRANDOMP_H_

#include <Security/SecBase.h>
#include <stdint.h>
#include <sys/types.h>
#include <Security/SecRandom.h>
#include <CoreFoundation/CoreFoundation.h>

#if defined(__cplusplus)
extern "C" {
#endif

/*!
 @function SecRandomCopyData
 @abstract Return count random bytes as a CFDataRef.
 @result Returns CFDataRef on success or NULL if something went wrong.
 */

CFDataRef
SecRandomCopyData(SecRandomRef rnd, size_t count)
__OSX_AVAILABLE_STARTING(__MAC_10_7, __IPHONE_NA);

#if defined(__cplusplus)
}
#endif

#endif /* !_SECURITY_SECRANDOM_H_ */
