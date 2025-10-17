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
#ifndef KeypressCommand_h
#define KeypressCommand_h

#include <wtf/Assertions.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(COCOA)

namespace WebCore {

struct KeypressCommand {
    KeypressCommand() { }

    explicit KeypressCommand(const String& commandName)
        : commandName(commandName)
    {
        ASSERT(isASCIILower(commandName[0U]));
    }

    KeypressCommand(const String& commandName, const String& text)
        : commandName(commandName)
        , text(text)
    {
        ASSERT(commandName == "insertText:"_s || text.isEmpty());
    }

    String commandName; // Actually, a selector name - it may have a trailing colon, and a name that can be different from an editor command name.
    String text;
};

} // namespace WebCore

#endif

#endif // KeypressCommand_h
