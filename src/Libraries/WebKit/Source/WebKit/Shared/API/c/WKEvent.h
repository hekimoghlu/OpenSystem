/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 15, 2022.
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
#ifndef WKEvent_h
#define WKEvent_h

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    kWKEventModifiersShiftKey = 1 << 0,
    kWKEventModifiersControlKey = 1 << 1,
    kWKEventModifiersAltKey = 1 << 2,
    kWKEventModifiersMetaKey = 1 << 3,
    kWKEventModifiersCapsLockKey = 1 << 4
};
typedef uint32_t WKEventModifiers;

enum {
    kWKEventMouseButtonLeftButton = 0,
    kWKEventMouseButtonMiddleButton = 1,
    kWKEventMouseButtonRightButton = 2,
    kWKEventMouseButtonNoButton = -2
};
typedef int32_t WKEventMouseButton;

#ifdef __cplusplus
}
#endif

#endif /* WKEvent_h */
