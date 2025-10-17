/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 4, 2023.
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

#include <JavaScriptCore/ConsoleTypes.h>
#include <JavaScriptCore/JSCJSValue.h>
#include <JavaScriptCore/Strong.h>
#include <JavaScriptCore/StrongInlines.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

enum class LoadableScriptErrorType : uint8_t {
    Fetch,
    CrossOriginLoad,
    MIMEType,
    Nosniff,
    FailedIntegrityCheck,
    Resolve,
    Script,
};

struct LoadableScriptConsoleMessage {
    JSC::MessageSource source;
    JSC::MessageLevel level;
    String message;
};

struct LoadableScriptError {
    LoadableScriptErrorType type;
    std::optional<LoadableScriptConsoleMessage> consoleMessage;
    JSC::Strong<JSC::Unknown> errorValue;
};

}
