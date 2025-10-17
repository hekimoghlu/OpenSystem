/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 7, 2022.
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
#include "CGWindowUtilities.h"

#if USE(CG) && PLATFORM(MAC)

#import <pal/cg/CoreGraphicsSoftLink.h>

namespace WebCore {

RetainPtr<CGImageRef> cgWindowListCreateImage(CGRect screenBounds, CGWindowListOption listOption, CGWindowID windowID, CGWindowImageOption imageOption)
{
    if (PAL::canLoad_CoreGraphics_CGWindowListCreateImage())
        return adoptCF(PAL::softLink_CoreGraphics_CGWindowListCreateImage(screenBounds, listOption, windowID, imageOption));

    return { };
}

}

#endif // USE(CG) && PLATFORM(MAC)
