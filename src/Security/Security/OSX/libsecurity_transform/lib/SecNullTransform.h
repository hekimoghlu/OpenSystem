/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 15, 2024.
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

#ifndef __SECNULLTRANSFORM__
#define __SECNULLTRANSFORM__


#ifdef COM_APPLE_SECURITY_ACTUALLY_BUILDING_LIBRARY
#include "SecTransform.h"
#else
#include <Security/SecTransform.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern const CFStringRef kSecNullTransformName;

typedef CFTypeRef SecNullTransformRef
API_DEPRECATED("SecTransform is no longer supported", macos(10.7, 13.0)) API_UNAVAILABLE(ios, tvos, watchos, macCatalyst);

SecNullTransformRef SecNullTransformCreate()
API_DEPRECATED("SecTransform is no longer supported", macos(10.7, 13.0)) API_UNAVAILABLE(ios, tvos, watchos, macCatalyst);

#ifdef __cplusplus
};
#endif

#endif
