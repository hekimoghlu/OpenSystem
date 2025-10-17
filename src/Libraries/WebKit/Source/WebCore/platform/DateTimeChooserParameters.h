/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 27, 2023.
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

#include "IntRect.h"
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct DateTimeChooserParameters {
    AtomString type;
    IntRect anchorRectInRootView;
    // Locale name for which the chooser should be localized.
    // This might be an invalid name because it comes from HTML lang attributes.
    AtomString locale;
    String currentValue;
    Vector<String> suggestionValues;
    Vector<String> localizedSuggestionValues;
    Vector<String> suggestionLabels;
    double minimum { 0 };
    double maximum { 0 };
    double step { 0 };
    double stepBase { 0 };
    bool required { false };
    bool isAnchorElementRTL { false };
    bool useDarkAppearance { false };
    bool hasSecondField { false };
    bool hasMillisecondField { false };
};

} // namespace WebCore
