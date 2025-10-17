/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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
#ifndef WKData_h
#define WKData_h

#include <WebKit/WKBase.h>

#include <stddef.h>

#ifdef __cplusplus
#include <span>

extern "C" {
#endif

WK_EXPORT WKTypeID WKDataGetTypeID(void);

WK_EXPORT WKDataRef WKDataCreate(const unsigned char* bytes, size_t size);

WK_EXPORT const unsigned char* WKDataGetBytes(WKDataRef data);
WK_EXPORT size_t WKDataGetSize(WKDataRef data);

#ifdef __cplusplus
}

WK_EXPORT std::span<const uint8_t> WKDataGetSpan(WKDataRef data);

#endif

#endif // WKData_h
