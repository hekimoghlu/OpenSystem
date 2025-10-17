/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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
#include "PlatformScreen.h"

#include "DestinationColorSpace.h"
#include "FloatRect.h"
#include "NotImplemented.h"

namespace WebCore {

int screenDepth(Widget*)
{
    notImplemented();
    return 24;
}

int screenDepthPerComponent(Widget*)
{
    notImplemented();
    return 8;
}

bool screenIsMonochrome(Widget*)
{
    notImplemented();
    return false;
}

bool screenHasInvertedColors()
{
    return false;
}

FloatRect screenRect(Widget*)
{
    notImplemented();
    return { 0, 0, 1920, 1080 };
}

FloatRect screenAvailableRect(Widget*)
{
    notImplemented();
    return { 0, 0, 1920, 1080 };
}

bool screenSupportsExtendedColor(Widget*)
{
    return false;
}

DestinationColorSpace screenColorSpace(Widget*)
{
    return DestinationColorSpace::SRGB();
}

} // namespace WebCore
