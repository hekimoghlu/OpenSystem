/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 5, 2024.
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

#if COMPILER(MSVC)
// Disable the "unreachable code" warning so we can compile the ASSERT_NOT_REACHED in the END_STATE macro.
#pragma warning(disable: 4702)
#endif

namespace WebCore {

inline bool isTokenizerWhitespace(UChar character)
{
    return character == ' ' || character == '\x0A' || character == '\x09' || character == '\x0C';
}

#define BEGIN_STATE(stateName)                                  \
    case stateName:                                             \
    stateName: {                                                \
        constexpr auto currentState = stateName;                \
        UNUSED_PARAM(currentState);

#define END_STATE()                                             \
        ASSERT_NOT_REACHED();                                   \
        break;                                                  \
    }

#define RETURN_IN_CURRENT_STATE(expression)                     \
    do {                                                        \
        m_state = currentState;                                 \
        return expression;                                      \
    } while (false)

// We use this macro when the HTML spec says "reconsume the current input character in the <mumble> state."
#define RECONSUME_IN(newState)                                  \
    do {                                                        \
        goto newState;                                          \
    } while (false)

// We use this macro when the HTML spec says "consume the next input character ... and switch to the <mumble> state."
#define ADVANCE_TO(newState)                                    \
    do {                                                        \
        if (!m_preprocessor.advance(source, isNullCharacterSkippingState(newState))) { \
            m_state = newState;                                 \
            return haveBufferedCharacterToken();                \
        }                                                       \
        character = m_preprocessor.nextInputCharacter();        \
        goto newState;                                          \
    } while (false)
#define ADVANCE_PAST_NON_NEWLINE_TO(newState)                   \
    do {                                                        \
        if (!m_preprocessor.advancePastNonNewline(source, isNullCharacterSkippingState(newState))) { \
            m_state = newState;                                 \
            return haveBufferedCharacterToken();                \
        }                                                       \
        character = m_preprocessor.nextInputCharacter();        \
        goto newState;                                          \
    } while (false)

// For more complex cases, caller consumes the characters first and then uses this macro.
#define SWITCH_TO(newState)                                     \
    do {                                                        \
        if (!m_preprocessor.peek(source, isNullCharacterSkippingState(newState))) { \
            m_state = newState;                                 \
            return haveBufferedCharacterToken();                \
        }                                                       \
        character = m_preprocessor.nextInputCharacter();        \
        goto newState;                                          \
    } while (false)

} // namespace WebCore
