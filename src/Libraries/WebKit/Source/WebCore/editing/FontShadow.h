/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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

#include "Color.h"
#include "FloatSize.h"
#include <wtf/RetainPtr.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(COCOA)
OBJC_CLASS NSShadow;
#endif

namespace WebCore {

struct FontShadow {
#if PLATFORM(COCOA)
    RetainPtr<NSShadow> createShadow() const;
#endif

    Color color;
    FloatSize offset;
    double blurRadius { 0 };
};

#if PLATFORM(COCOA)
WEBCORE_EXPORT FontShadow fontShadowFromNSShadow(NSShadow *);
#endif

WEBCORE_EXPORT String serializationForCSS(const FontShadow&);

} // namespace WebCore
