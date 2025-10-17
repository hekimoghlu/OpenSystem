/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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
#include "config.h"

#if USE(CG)

#include <pal/spi/cg/CoreGraphicsSPI.h>
#include <wtf/SoftLinking.h>

#if HAVE(CG_CONTEXT_SET_OWNER_IDENTITY)
#include <mach/mach_types.h>
#endif

SOFT_LINK_FRAMEWORK_FOR_SOURCE_WITH_EXPORT(PAL, CoreGraphics, PAL_EXPORT)

#if HAVE(CG_CONTEXT_SET_OWNER_IDENTITY)
SOFT_LINK_FUNCTION_FOR_SOURCE_WITH_EXPORT(PAL, CoreGraphics, CGContextSetOwnerIdentity, void, (CGContextRef context, task_id_token_t owner), (context, owner), PAL_EXPORT)
#endif

#if PLATFORM(MAC)
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, CoreGraphics, CGWindowListCreateImage, CGImageRef, (CGRect screenBounds, CGWindowListOption listOption, CGWindowID windowID, CGWindowImageOption imageOption), (screenBounds, listOption, windowID, imageOption), PAL_EXPORT)
#endif

#if HAVE(IOSURFACE)
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE_WITH_EXPORT(PAL, CoreGraphics, CGIOSurfaceContextInvalidateSurface, void, (CGContextRef context), (context), PAL_EXPORT)
#endif

#endif
