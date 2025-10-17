/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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

namespace WebCore {

class Element;
class StyleSheetContents;

namespace Style {

class RuleSet;

class UserAgentStyle {
public:
    static RuleSet* defaultStyle;
    static RuleSet* defaultQuirksStyle;
    static RuleSet* defaultPrintStyle;
    static unsigned defaultStyleVersion;

    static StyleSheetContents* defaultStyleSheet;
    static StyleSheetContents* quirksStyleSheet;
    static StyleSheetContents* svgStyleSheet;
    static StyleSheetContents* mathMLStyleSheet;
    static StyleSheetContents* mediaQueryStyleSheet;
    static StyleSheetContents* horizontalFormControlsStyleSheet;
    static StyleSheetContents* htmlSwitchControlStyleSheet;
    static StyleSheetContents* popoverStyleSheet;
    static StyleSheetContents* counterStylesStyleSheet;
    static StyleSheetContents* viewTransitionsStyleSheet;
#if ENABLE(FULLSCREEN_API)
    static StyleSheetContents* fullscreenStyleSheet;
#endif
#if ENABLE(SERVICE_CONTROLS)
    static StyleSheetContents* imageControlsStyleSheet;
#endif
#if ENABLE(ATTACHMENT_ELEMENT)
    static StyleSheetContents* attachmentStyleSheet;
#endif

    static void initDefaultStyleSheet();
    static void ensureDefaultStyleSheetsForElement(const Element&);

private:
    static void addToDefaultStyle(StyleSheetContents&);
};

} // namespace Style
} // namespace WebCore
