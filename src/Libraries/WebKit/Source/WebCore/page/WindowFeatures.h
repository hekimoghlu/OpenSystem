/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 1, 2022.
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

#include "DisabledAdaptations.h"
#include "FloatRect.h"
#include <wtf/Function.h>
#include <wtf/OptionSet.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct WindowFeatures {
    bool hasAdditionalFeatures { false };

    std::optional<float> x;
    std::optional<float> y;
    std::optional<float> width;
    std::optional<float> height;

    std::optional<bool> popup;
    std::optional<bool> menuBarVisible;
    std::optional<bool> statusBarVisible;
    std::optional<bool> toolBarVisible;
    std::optional<bool> locationBarVisible;
    std::optional<bool> scrollbarsVisible;
    std::optional<bool> resizable;

#if PLATFORM(GTK)
    FloatRect oldWindowRect { };
#endif

    std::optional<bool> fullscreen;
    std::optional<bool> dialog;
    std::optional<bool> noopener { std::nullopt };
    std::optional<bool> noreferrer { std::nullopt };

    Vector<String> additionalFeatures { };

    bool wantsNoOpener() const { return (noopener && *noopener) || (noreferrer && *noreferrer); }
    bool wantsNoReferrer() const { return (noreferrer && *noreferrer); }

    // Follow the HTML standard on how to parse the window features indicated here:
    // https://html.spec.whatwg.org/multipage/nav-history-apis.html#apis-for-creating-and-navigating-browsing-contexts-by-name
    bool wantsPopup() const
    {
        // If the WindowFeatures string contains nothing more than noopener and noreferrer we
        // consider the string to be empty and thus return false based on the algorithm above.
        if (!hasAdditionalFeatures
            && !x
            && !y
            && !width
            && !height
            && !popup
            && !menuBarVisible
            && !statusBarVisible
            && !toolBarVisible
            && !locationBarVisible
            && !scrollbarsVisible
            && !resizable)
            return false;

        // If popup is defined, return its value as a boolean.
        if (popup)
            return *popup;

        // If location (default to false) and toolbar (default to false) are false return true.
        if ((!locationBarVisible || !*locationBarVisible) && (!toolBarVisible || !*toolBarVisible))
            return true;

        // If menubar (default to false) is false return true.
        if (!menuBarVisible || !*menuBarVisible)
            return true;

        // If resizable (default to true) is false return true.
        if (resizable && !*resizable)
            return true;

        // If scrollbars (default to false) is false return false.
        if (!scrollbarsVisible || !*scrollbarsVisible)
            return true;

        // If status (default to false) is false return true.
        if (!statusBarVisible || !*statusBarVisible)
            return true;

        return false;
    }
};

WindowFeatures parseWindowFeatures(StringView windowFeaturesString);
WindowFeatures parseDialogFeatures(StringView dialogFeaturesString, const FloatRect& screenAvailableRect);
OptionSet<DisabledAdaptations> parseDisabledAdaptations(StringView);

enum class FeatureMode { Window, Viewport };
void processFeaturesString(StringView features, FeatureMode, const Function<void(StringView type, StringView value)>& callback);

} // namespace WebCore
