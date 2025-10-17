/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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

#include "FloatPoint.h"
#include "TextIndicator.h"

#if PLATFORM(COCOA)
#include "AttributedString.h"
#include <wtf/RetainPtr.h>
OBJC_CLASS NSDictionary;
#endif

namespace WebCore {

#if PLATFORM(COCOA)
struct DictionaryPopupInfoCocoa {
    AttributedString attributedString;
};
#endif

struct DictionaryPopupInfo {
    FloatPoint origin;
    TextIndicatorData textIndicator;

    // FIXME: This can be a plain string (and cross-platform) once all clients
    // vend fully-formed TextIndicatorData. Legacy PDFPlugin is the last client.
#if PLATFORM(COCOA)
    DictionaryPopupInfoCocoa platformData;
#endif
};

} // namespace WebCore
