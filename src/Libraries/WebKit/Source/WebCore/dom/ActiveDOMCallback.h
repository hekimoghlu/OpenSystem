/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 25, 2024.
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

#include "ContextDestructionObserver.h"

namespace JSC {
class AbstractSlotVisitor;
class SlotVisitor;
}

namespace WebCore {

class ScriptExecutionContext;

// A base class that prevents binding callbacks from executing when
// active dom objects are stopped or suspended.
//
// Should only be created, used, and destroyed on the script execution
// context thread.
class ActiveDOMCallback : public ContextDestructionObserver {
public:
    WEBCORE_EXPORT ActiveDOMCallback(ScriptExecutionContext*);
    WEBCORE_EXPORT virtual ~ActiveDOMCallback();

    WEBCORE_EXPORT bool canInvokeCallback() const;

    WEBCORE_EXPORT bool activeDOMObjectsAreSuspended() const;
    WEBCORE_EXPORT bool activeDOMObjectAreStopped() const;
    
    virtual void visitJSFunction(JSC::AbstractSlotVisitor&) { }
    virtual void visitJSFunction(JSC::SlotVisitor&) { }
};

} // namespace WebCore
