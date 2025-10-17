/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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

#include <WebCore/AutocapitalizeTypes.h>
#include <WebCore/InputMode.h>
#include <wtf/OptionSet.h>

namespace IPC {
class Decoder;
class Encoder;
}

namespace WebCore {
class HTMLInputElement;
}

namespace WebKit {

struct InputMethodState {
    enum class Purpose : uint8_t {
        FreeForm,
        Digits,
        Number,
        Phone,
        Url,
        Email,
        Password
    };

    enum class Hint : uint8_t {
        None = 0,
        Spellcheck = 1 << 0,
        Lowercase = 1 << 1,
        UppercaseChars = 1 << 2,
        UppercaseWords = 1 << 3,
        UppercaseSentences = 1 << 4,
        InhibitOnScreenKeyboard = 1 << 5
    };

    void setPurposeOrHintForInputMode(WebCore::InputMode);
    void setPurposeForInputElement(WebCore::HTMLInputElement&);
    void addHintsForAutocapitalizeType(WebCore::AutocapitalizeType);

    friend bool operator==(const InputMethodState&, const InputMethodState&) = default;

    Purpose purpose { Purpose::FreeForm };
    OptionSet<Hint> hints;
};

} // namespace WebKit
