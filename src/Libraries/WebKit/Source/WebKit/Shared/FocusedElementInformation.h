/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 7, 2023.
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

#include "ArgumentCoders.h"
#include "ColorControlSupportsAlpha.h"
#include "IdentifierTypes.h"
#include <WebCore/AutocapitalizeTypes.h>
#include <WebCore/Autofill.h>
#include <WebCore/Color.h>
#include <WebCore/ElementContext.h>
#include <WebCore/EnterKeyHint.h>
#include <WebCore/FrameIdentifier.h>
#include <WebCore/InputMode.h>
#include <WebCore/IntRect.h>
#include <WebCore/ScrollTypes.h>
#include <wtf/URL.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

enum class InputType : uint8_t {
    None,
    ContentEditable,
    Text,
    Password,
    TextArea,
    Search,
    Email,
    URL,
    Phone,
    Number,
    NumberPad,
    Date,
    DateTimeLocal,
    Month,
    Week,
    Time,
    Select,
    Drawing,
    Color
};

#if PLATFORM(IOS_FAMILY)
struct OptionItem {
    OptionItem() = default;

    OptionItem(const String& text, bool isGroup, bool selected, bool disabled, int parentID)
        : text(text)
        , isGroup(isGroup)
        , isSelected(selected)
        , disabled(disabled)
        , parentGroupID(parentID)
    {
    }

    String text;
    bool isGroup { false };
    bool isSelected { false };
    bool disabled { false };
    int parentGroupID { 0 };
};

struct FocusedElementInformation {
    WebCore::IntRect interactionRect;
    WebCore::ElementContext elementContext;
    WebCore::IntPoint lastInteractionLocation;
    double minimumScaleFactor { -INFINITY };
    double maximumScaleFactor { INFINITY };
    double maximumScaleFactorIgnoringAlwaysScalable { INFINITY };
    double nodeFontSize { 0 };
    bool hasNextNode { false };
    WebCore::IntRect nextNodeRect;
    bool hasPreviousNode { false };
    WebCore::IntRect previousNodeRect;
    bool isAutocorrect { false };
    bool isRTL { false };
    bool isMultiSelect { false };
    bool isReadOnly {false };
    bool allowsUserScaling { false };
    bool allowsUserScalingIgnoringAlwaysScalable { false };
    bool insideFixedPosition { false };
    bool hasPlainText { false };
    WebCore::AutocapitalizeType autocapitalizeType { WebCore::AutocapitalizeType::Default };
    InputType elementType { InputType::None };
    WebCore::InputMode inputMode { WebCore::InputMode::Unspecified };
    WebCore::EnterKeyHint enterKeyHint { WebCore::EnterKeyHint::Unspecified };
    String formAction;
    Vector<OptionItem> selectOptions;
    int selectedIndex { -1 };
    String value;
    double valueAsNumber { 0 };
    String title;
    bool acceptsAutofilledLoginCredentials { false };
    bool isAutofillableUsernameField { false };
    URL representingPageURL;
    WebCore::AutofillFieldName autofillFieldName { WebCore::AutofillFieldName::None };
    WebCore::NonAutofillCredentialType nonAutofillCredentialType { WebCore::NonAutofillCredentialType::None };
    String placeholder;
    String label;
    String ariaLabel;
    bool hasSuggestions { false };
    bool isFocusingWithDataListDropdown { false };
    WebCore::Color colorValue;
    ColorControlSupportsAlpha supportsAlpha { ColorControlSupportsAlpha::No };
    Vector<WebCore::Color> suggestedColors;
    bool hasEverBeenPasswordField { false };
    bool shouldSynthesizeKeyEventsForEditing { false };
    bool isSpellCheckingEnabled { true };
    bool isWritingSuggestionsEnabled { false };
    bool shouldAvoidResizingWhenInputViewBoundsChange { false };
    bool shouldAvoidScrollingWhenFocusedContentIsVisible { false };
    bool shouldUseLegacySelectPopoverDismissalBehaviorInDataActivation { false };
    bool isFocusingWithValidationMessage { false };
    bool preventScroll { false };

    FocusedElementInformationIdentifier identifier;
    Markable<WebCore::ScrollingNodeID> containerScrollingNodeID;

    Markable<WebCore::FrameIdentifier> frameID;
};
#endif

} // namespace WebKit
