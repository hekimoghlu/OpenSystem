/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 13, 2025.
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

#include "Label.h"

namespace JSC {

class Identifier;

class LabelScope {
WTF_MAKE_NONCOPYABLE(LabelScope);
public:
    enum Type { Loop, Switch, NamedLabel };

    LabelScope(Type type, const Identifier* name, int scopeDepth, Ref<Label>&& breakTarget, RefPtr<Label>&& continueTarget)
        : m_refCount(0)
        , m_type(type)
        , m_name(name)
        , m_scopeDepth(scopeDepth)
        , m_breakTarget(WTFMove(breakTarget))
        , m_continueTarget(WTFMove(continueTarget))
    {
    }

    Label& breakTarget() const { return m_breakTarget.get(); }
    Label* continueTarget() const { return m_continueTarget.get(); }

    Type type() const { return m_type; }
    const Identifier* name() const { return m_name; }
    int scopeDepth() const { return m_scopeDepth; }

    void ref() { ++m_refCount; }
    void deref()
    {
        --m_refCount;
        ASSERT(m_refCount >= 0);
    }
    int refCount() const { return m_refCount; }
    bool hasOneRef() const { return m_refCount == 1; }

    bool breakTargetMayBeBound() const
    {
        if (!hasOneRef())
            return true;
        if (!m_breakTarget->hasOneRef())
            return true;
        return m_breakTarget->isBound();
    }

private:
    int m_refCount;
    Type m_type;
    const Identifier* m_name;
    int m_scopeDepth;
    Ref<Label> m_breakTarget;
    RefPtr<Label> m_continueTarget;
};

} // namespace JSC
