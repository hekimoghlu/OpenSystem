/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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
#import "FontShadow.h"

#import "ColorCocoa.h"
#import "ColorMac.h"

#if PLATFORM(IOS_FAMILY)
#import <pal/ios/UIKitSoftLink.h>
#endif

namespace WebCore {

RetainPtr<NSShadow> FontShadow::createShadow() const
{
#if USE(APPKIT)
    auto shadow = adoptNS([NSShadow new]);
#elif PLATFORM(IOS_FAMILY)
    auto shadow = adoptNS([PAL::getNSShadowClass() new]);
#endif
    [shadow setShadowColor:cocoaColor(color).get()];
    [shadow setShadowOffset:offset];
    [shadow setShadowBlurRadius:blurRadius];
    return shadow;
}

FontShadow fontShadowFromNSShadow(NSShadow *shadow)
{
    return {
        colorFromCocoaColor(shadow.shadowColor),
        FloatSize(shadow.shadowOffset),
        shadow.shadowBlurRadius,
    };
}

}
