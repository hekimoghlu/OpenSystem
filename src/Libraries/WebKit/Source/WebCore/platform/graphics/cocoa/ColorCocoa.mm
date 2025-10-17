/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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
#import "config.h"
#import "ColorCocoa.h"

#if PLATFORM(IOS_FAMILY)
#import <pal/ios/UIKitSoftLink.h>
#import <pal/spi/ios/UIKitSPI.h>
#endif

namespace WebCore {

#if PLATFORM(IOS_FAMILY)

RetainPtr<UIColor> cocoaColor(const Color& color)
{
    return [PAL::getUIColorClass() _disambiguated_due_to_CIImage_colorWithCGColor:cachedCGColor(color).get()];
}

#endif

RetainPtr<CocoaColor> cocoaColorOrNil(const Color& color)
{
    return color.isValid() ? cocoaColor(color) : nil;
}

} // namespace WebCore
