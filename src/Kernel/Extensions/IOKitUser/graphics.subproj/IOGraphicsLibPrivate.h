/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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
#ifndef _IOKIT_IOGRAPHICSLIBPRIVATE_H
#define _IOKIT_IOGRAPHICSLIBPRIVATE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <IOKit/IOKitLib.h>
#include <IOKit/graphics/IOGraphicsLib.h>

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#define kIOFBModeResolutionNameKey      "name"
#define kIOFBModeRefreshNameKey         "refresh"

extern CFDictionaryRef
IOFBCreateModeInfoDictionary(
        io_service_t                    framebuffer,
        IOOptionBits                    options,
        IODisplayModeID                 displayMode,
        IODisplayModeInformation *      info);

kern_return_t
IOFBGetDisplayModeTimingInformation( io_connect_t connect,
        IODisplayModeID               displayMode,
        IODetailedTimingInformation * out );

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifdef __cplusplus
}
#endif

#endif /* ! _IOKIT_IOGRAPHICSLIBPRIVATE_H */
