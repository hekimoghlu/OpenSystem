/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 14, 2024.
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

#include "ArityCheckMode.h"
#include "CallMode.h"
#include "JSCPtrTag.h"
#include <wtf/CodePtr.h>
#include <wtf/SentinelLinkedList.h>

namespace JSC {

class CodeBlock;
class JSCell;
class VM;

class CallSlot {
public:
    JSCell* m_calleeOrExecutable { nullptr };
    uint32_t m_count { 0 };
    uint8_t m_index { 0 };
    ArityCheckMode m_arityCheckMode { MustCheckArity };
    CodePtr<JSEntryPtrTag> m_target;
    CodeBlock* m_codeBlock { nullptr }; // This is weakly held. And cleared whenever m_target is changed.

    static constexpr ptrdiff_t offsetOfCalleeOrExecutable() { return OBJECT_OFFSETOF(CallSlot, m_calleeOrExecutable); }
    static constexpr ptrdiff_t offsetOfCount() { return OBJECT_OFFSETOF(CallSlot, m_count); }
    static constexpr ptrdiff_t offsetOfTarget() { return OBJECT_OFFSETOF(CallSlot, m_target); }
    static constexpr ptrdiff_t offsetOfCodeBlock() { return OBJECT_OFFSETOF(CallSlot, m_codeBlock); }
};
static_assert(sizeof(CallSlot) <= 32, "This should be small enough to keep iteration of vector in polymorphic call fast");

class CallLinkInfoBase : public BasicRawSentinelNode<CallLinkInfoBase> {
public:
    enum class CallSiteType : uint8_t {
        CallLinkInfo,
        PolymorphicCallNode,
#if ENABLE(JIT)
        DirectCall,
#endif
        CachedCall,
    };

    enum CallType : uint8_t {
        None,
        Call,
        CallVarargs,
        Construct,
        ConstructVarargs,
        TailCall,
        TailCallVarargs,
        DirectCall,
        DirectConstruct,
        DirectTailCall
    };

    enum class UseDataIC : bool { No, Yes };

    static CallMode callModeFor(CallType callType)
    {
        switch (callType) {
        case Call:
        case CallVarargs:
        case DirectCall:
            return CallMode::Regular;
        case TailCall:
        case TailCallVarargs:
        case DirectTailCall:
            return CallMode::Tail;
        case Construct:
        case ConstructVarargs:
        case DirectConstruct:
            return CallMode::Construct;
        case None:
            RELEASE_ASSERT_NOT_REACHED();
        }

        RELEASE_ASSERT_NOT_REACHED();
    }

    explicit CallLinkInfoBase(CallSiteType callSiteType)
        : m_callSiteType(callSiteType)
    {
    }

    ~CallLinkInfoBase()
    {
        if (isOnList())
            remove();
    }

    CallSiteType callSiteType() const { return m_callSiteType; }

    void unlinkOrUpgrade(VM&, CodeBlock*, CodeBlock*);

private:
    CallSiteType m_callSiteType;
};

} // namespace JSC
