/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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
#pragma once

#import "Color.h"

#if USE(APPKIT)
OBJC_CLASS NSColor;
#endif

#if PLATFORM(IOS_FAMILY)
OBJC_CLASS UIColor;
#endif

namespace WebCore {

class Color;

#if USE(APPKIT)
using CocoaColor = NSColor;
#endif

#if PLATFORM(IOS_FAMILY)
using CocoaColor = UIColor;
#endif

WEBCORE_EXPORT RetainPtr<CocoaColor> cocoaColor(const Color&);
WEBCORE_EXPORT RetainPtr<CocoaColor> cocoaColorOrNil(const Color&); // Returns nil if the color is invalid.
WEBCORE_EXPORT Color colorFromCocoaColor(CocoaColor *);

} // namespace WebCore
