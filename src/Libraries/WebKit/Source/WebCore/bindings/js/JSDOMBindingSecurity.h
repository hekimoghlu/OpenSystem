/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 20, 2024.
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

#include "ExceptionOr.h"
#include <wtf/Forward.h>

namespace JSC {
class CallFrame;
class JSGlobalObject;
}

namespace WebCore {

class DOMWindow;
class LocalDOMWindow;
class LocalFrame;
class Node;

void printErrorMessageForFrame(LocalFrame*, const String& message);

enum SecurityReportingOption { DoNotReportSecurityError, LogSecurityError, ThrowSecurityError };

namespace BindingSecurity {

template<typename T> T* checkSecurityForNode(JSC::JSGlobalObject&, T&);
template<typename T> T* checkSecurityForNode(JSC::JSGlobalObject&, T*);
template<typename T> ExceptionOr<T*> checkSecurityForNode(JSC::JSGlobalObject&, ExceptionOr<T*>&&);
template<typename T> ExceptionOr<T*> checkSecurityForNode(JSC::JSGlobalObject&, ExceptionOr<T&>&&);

bool shouldAllowAccessToDOMWindow(JSC::JSGlobalObject*, LocalDOMWindow&, SecurityReportingOption = LogSecurityError);
bool shouldAllowAccessToDOMWindow(JSC::JSGlobalObject&, LocalDOMWindow&, String& message);
bool shouldAllowAccessToDOMWindow(JSC::JSGlobalObject*, LocalDOMWindow*, SecurityReportingOption = LogSecurityError);
bool shouldAllowAccessToDOMWindow(JSC::JSGlobalObject&, LocalDOMWindow*, String& message);
bool shouldAllowAccessToDOMWindow(JSC::JSGlobalObject*, DOMWindow&, SecurityReportingOption = LogSecurityError);
bool shouldAllowAccessToDOMWindow(JSC::JSGlobalObject&, DOMWindow&, String& message);
bool shouldAllowAccessToDOMWindow(JSC::JSGlobalObject*, DOMWindow*, SecurityReportingOption = LogSecurityError);
bool shouldAllowAccessToDOMWindow(JSC::JSGlobalObject&, DOMWindow*, String& message);
bool shouldAllowAccessToFrame(JSC::JSGlobalObject*, LocalFrame*, SecurityReportingOption = LogSecurityError);
bool shouldAllowAccessToFrame(JSC::JSGlobalObject&, LocalFrame&, String& message);
bool shouldAllowAccessToNode(JSC::JSGlobalObject&, Node*);

}

template<typename T> inline T* BindingSecurity::checkSecurityForNode(JSC::JSGlobalObject& lexicalGlobalObject, T& node)
{
    return shouldAllowAccessToNode(lexicalGlobalObject, &node) ? &node : nullptr;
}

template<typename T> inline T* BindingSecurity::checkSecurityForNode(JSC::JSGlobalObject& lexicalGlobalObject, T* node)
{
    return shouldAllowAccessToNode(lexicalGlobalObject, node) ? node : nullptr;
}

template<typename T> inline ExceptionOr<T*> BindingSecurity::checkSecurityForNode(JSC::JSGlobalObject& lexicalGlobalObject, ExceptionOr<T*>&& value)
{
    if (value.hasException())
        return value.releaseException();
    return checkSecurityForNode(lexicalGlobalObject, value.releaseReturnValue());
}

template<typename T> inline ExceptionOr<T*> BindingSecurity::checkSecurityForNode(JSC::JSGlobalObject& lexicalGlobalObject, ExceptionOr<T&>&& value)
{
    if (value.hasException())
        return value.releaseException();
    return checkSecurityForNode(lexicalGlobalObject, value.releaseReturnValue());
}

} // namespace WebCore
