/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 19, 2024.
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
#ifndef SecPaddingConfigurations_h
#define SecPaddingConfigurations_h

#include <CoreFoundation/CoreFoundation.h>

typedef CF_ENUM(uint8_t, SecPaddingType) {
	SecPaddingTypeMMCS CF_ENUM_AVAILABLE(10_13, 11_0) = 0,
} CF_ENUM_AVAILABLE(10_13, 11_0);

/*!
 @function
 @abstract   Compute the padding size given the size of the content
 @param      type Type of content to be protected
 @param      size size before padding
 @param      error Output parameter to a CFErrorRef
 @result     number of bytes to add to the message. Only returns a negative value on SecPaddingType mismatch with a CFError assigned
 */
int64_t SecPaddingCompute(SecPaddingType type, uint32_t size, CFErrorRef *error)
__OSX_AVAILABLE(10.13) __IOS_AVAILABLE(11.0) __TVOS_AVAILABLE(11.0) __WATCHOS_AVAILABLE(4.0);

#endif
