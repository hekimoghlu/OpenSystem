/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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
#include "PaymentRequestUtilities.h"

#if ENABLE(PAYMENT_REQUEST)

namespace WebCore {

// Implements the "valid decimal monetary value" validity checker
// https://www.w3.org/TR/payment-request/#dfn-valid-decimal-monetary-value
bool isValidDecimalMonetaryValue(StringView value)
{
    enum class State {
        Start,
        Sign,
        Digit,
        Dot,
        DotDigit,
    };

    auto state = State::Start;
    for (auto character : value.codeUnits()) {
        switch (state) {
        case State::Start:
            if (character == '-') {
                state = State::Sign;
                break;
            }

            if (isASCIIDigit(character)) {
                state = State::Digit;
                break;
            }

            return false;

        case State::Sign:
            if (isASCIIDigit(character)) {
                state = State::Digit;
                break;
            }

            return false;

        case State::Digit:
            if (character == '.') {
                state = State::Dot;
                break;
            }

            if (isASCIIDigit(character)) {
                state = State::Digit;
                break;
            }

            return false;

        case State::Dot:
            if (isASCIIDigit(character)) {
                state = State::DotDigit;
                break;
            }

            return false;

        case State::DotDigit:
            if (isASCIIDigit(character)) {
                state = State::DotDigit;
                break;
            }

            return false;
        }
    }

    return state == State::Digit || state == State::DotDigit;
}

} // namespace WebCore

#endif // ENABLE(PAYMENT_REQUEST)
