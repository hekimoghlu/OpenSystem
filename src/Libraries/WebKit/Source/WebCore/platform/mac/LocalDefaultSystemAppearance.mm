/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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
#import "LocalDefaultSystemAppearance.h"

#if USE(APPKIT)

#import "ColorMac.h"

#import <AppKit/NSAppearance.h>
#import <pal/spi/mac/NSAppearanceSPI.h>

namespace WebCore {

LocalDefaultSystemAppearance::LocalDefaultSystemAppearance(bool useDarkAppearance, const Color& tintColor)
{
    m_savedSystemAppearance = [NSAppearance currentDrawingAppearance];
    m_usingDarkAppearance = useDarkAppearance;

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    NSAppearance *appearance = [NSAppearance appearanceNamed:m_usingDarkAppearance ? NSAppearanceNameDarkAqua : NSAppearanceNameAqua];

    if (tintColor.isValid())
        appearance = [appearance appearanceByApplyingTintColor:cocoaColor(tintColor).get()];

    [NSAppearance setCurrentAppearance:appearance];
ALLOW_DEPRECATED_DECLARATIONS_END
}

LocalDefaultSystemAppearance::~LocalDefaultSystemAppearance()
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    [NSAppearance setCurrentAppearance:m_savedSystemAppearance.get()];
ALLOW_DEPRECATED_DECLARATIONS_END
}

}

#endif // USE(APPKIT)
