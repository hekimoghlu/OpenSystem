/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 11, 2023.
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

#include <wtf/Forward.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class ScriptExecutionContext;

class ContextDestructionObserver {
public:
    WEBCORE_EXPORT virtual void contextDestroyed();

    ScriptExecutionContext* scriptExecutionContext() const; // Defined in ContextDestructionObserverInlines.h.
    RefPtr<ScriptExecutionContext> protectedScriptExecutionContext() const;

protected:
    WEBCORE_EXPORT explicit ContextDestructionObserver(ScriptExecutionContext*);
    WEBCORE_EXPORT virtual ~ContextDestructionObserver();
    void observeContext(ScriptExecutionContext*);

private:
    WeakPtr<ScriptExecutionContext> m_scriptExecutionContext;
};

} // namespace WebCore
