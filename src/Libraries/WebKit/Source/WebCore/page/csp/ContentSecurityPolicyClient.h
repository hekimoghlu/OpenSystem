/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 12, 2023.
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

#include "SecurityPolicyViolationEvent.h"
#include <JavaScriptCore/ConsoleTypes.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class FormData;

struct CSPInfo {
    String documentURI;
    String sourceFile;
    String sample;
    int lineNumber { 0 };
    int columnNumber { 0 };
};

struct WEBCORE_EXPORT ContentSecurityPolicyClient {
    // An inline function cannot be the first non-abstract virtual function declared
    // in the class as it results in the vtable being generated as a weak symbol.
    // This hurts performance (in Mac OS X at least, when loading frameworks), so we
    // don't want to do it in WebKit.
    virtual void addConsoleMessage(MessageSource, MessageLevel, const String&, unsigned long requestIdentifier = 0) = 0;

    virtual ~ContentSecurityPolicyClient() = default;

    virtual void enqueueSecurityPolicyViolationEvent(SecurityPolicyViolationEventInit&&) = 0;
};

} // namespace WebCore
