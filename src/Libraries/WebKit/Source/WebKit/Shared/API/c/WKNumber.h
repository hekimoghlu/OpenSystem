/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 5, 2022.
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
#ifndef WKNumber_h
#define WKNumber_h

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

/* WKBoolean */
WK_EXPORT WKTypeID WKBooleanGetTypeID(void);
WK_EXPORT WKBooleanRef WKBooleanCreate(bool value);
WK_EXPORT bool WKBooleanGetValue(WKBooleanRef booleanRef);

/* WKDouble */
WK_EXPORT WKTypeID WKDoubleGetTypeID(void);
WK_EXPORT WKDoubleRef WKDoubleCreate(double value);
WK_EXPORT double WKDoubleGetValue(WKDoubleRef doubleRef);

/* WKUInt64 */
WK_EXPORT WKTypeID WKUInt64GetTypeID(void);
WK_EXPORT WKUInt64Ref WKUInt64Create(uint64_t value);
WK_EXPORT uint64_t WKUInt64GetValue(WKUInt64Ref integerRef);

#ifdef __cplusplus
}
#endif

#endif /* WKNumber_h */
