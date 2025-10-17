/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 10, 2022.
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

#include "CallFrame.h"

namespace JSC {

class DebuggerEvalEnabler {
public:
    enum class Mode {
        EvalOnCurrentCallFrame,
        EvalOnGlobalObjectAtDebuggerEntry,
    };

    DebuggerEvalEnabler(JSGlobalObject* globalObject, Mode mode = Mode::EvalOnCurrentCallFrame)
        : m_globalObject(globalObject)
#if ASSERT_ENABLED
        , m_mode(mode)
#endif
        
    {
        UNUSED_PARAM(mode);
        if (globalObject) {
            m_evalWasDisabled = !globalObject->evalEnabled();
            m_trustedTypesWereRequired = globalObject->requiresTrustedTypes();
            if (m_evalWasDisabled)
                globalObject->setEvalEnabled(true, globalObject->evalDisabledErrorMessage());
            if (m_trustedTypesWereRequired)
                globalObject->setRequiresTrustedTypes(false);
#if ASSERT_ENABLED
            if (m_mode == Mode::EvalOnGlobalObjectAtDebuggerEntry)
                globalObject->setGlobalObjectAtDebuggerEntry(globalObject);
#endif
        }
    }

    ~DebuggerEvalEnabler()
    {
        if (m_globalObject) {
            JSGlobalObject* globalObject = m_globalObject;
            if (m_evalWasDisabled)
                globalObject->setEvalEnabled(false, globalObject->evalDisabledErrorMessage());
            if (m_trustedTypesWereRequired)
                globalObject->setRequiresTrustedTypes(true);
#if ASSERT_ENABLED
            if (m_mode == Mode::EvalOnGlobalObjectAtDebuggerEntry)
                globalObject->setGlobalObjectAtDebuggerEntry(nullptr);
#endif
        }
    }

private:
    JSGlobalObject* const m_globalObject;
    bool m_evalWasDisabled { false };
    bool m_trustedTypesWereRequired { false };
#if ASSERT_ENABLED
    DebuggerEvalEnabler::Mode m_mode;
#endif
};

} // namespace JSC
