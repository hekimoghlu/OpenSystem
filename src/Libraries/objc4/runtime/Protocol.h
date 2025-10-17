/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 21, 2025.
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
	Protocol.h
	Copyright 1991-1996 NeXT Software, Inc.
*/

#ifndef _OBJC_PROTOCOL_H_
#define _OBJC_PROTOCOL_H_

#if !__OBJC__

// typedef Protocol is here:
#include <objc/runtime.h>


#else

#include <objc/NSObject.h>

// All methods of class Protocol are unavailable. 
// Use the functions in objc/runtime.h instead.

OBJC_AVAILABLE(10.0, 2.0, 9.0, 1.0, 2.0)
@interface Protocol : NSObject
@end

#endif

#endif /* _OBJC_PROTOCOL_H_ */
