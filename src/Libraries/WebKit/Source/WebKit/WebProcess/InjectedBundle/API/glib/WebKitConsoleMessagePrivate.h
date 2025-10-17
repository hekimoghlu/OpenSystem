/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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

#include "WebKitConsoleMessage.h"
#include <JavaScriptCore/ConsoleTypes.h>
#include <wtf/text/CString.h>
#include <wtf/text/WTFString.h>

struct _WebKitConsoleMessage {
    _WebKitConsoleMessage(JSC::MessageSource source, JSC::MessageLevel level, const String& message, unsigned lineNumber, const String& sourceID)
        : source(source)
        , level(level)
        , message(message.utf8())
        , lineNumber(lineNumber)
        , sourceID(sourceID.utf8())
    {
    }

    _WebKitConsoleMessage(WebKitConsoleMessage* consoleMessage)
        : source(consoleMessage->source)
        , level(consoleMessage->level)
        , message(consoleMessage->message)
        , lineNumber(consoleMessage->lineNumber)
        , sourceID(consoleMessage->sourceID)
    {
    }

    JSC::MessageSource source;
    JSC::MessageLevel level;
    CString message;
    unsigned lineNumber;
    CString sourceID;
};
