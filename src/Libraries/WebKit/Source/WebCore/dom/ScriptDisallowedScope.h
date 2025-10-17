/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 26, 2024.
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

#include "ContainerNode.h"
#include <wtf/MainThread.h>
#include <wtf/RuntimeApplicationChecks.h>

#if PLATFORM(IOS_FAMILY)
#include "WebCoreThread.h"
#endif

namespace WebCore {

class ScriptDisallowedScope {
public:
    // This variant is expensive. Use ScriptDisallowedScope::InMainThread whenever possible.
    ScriptDisallowedScope()
    {
        if (!isMainThread())
            return;
        ++s_count;
    }

    ScriptDisallowedScope(const ScriptDisallowedScope&)
        : ScriptDisallowedScope()
    {
    }

    ~ScriptDisallowedScope()
    {
        if (!isMainThread())
            return;
        ASSERT(s_count);
        s_count--;
    }

    ScriptDisallowedScope& operator=(const ScriptDisallowedScope&)
    {
        return *this;
    }

    static bool isScriptAllowedInMainThread()
    {
        return !isInWebProcess() || !isMainThread() || !s_count;
    }

    class InMainThread {
    public:
        InMainThread()
        {
            ASSERT(isMainThread());
            ++s_count;
        }

        ~InMainThread()
        {
            ASSERT(isMainThread());
            ASSERT(s_count);
            --s_count;
        }

        // Don't enable this assertion in release since it's O(n).
        // Release asserts in canExecuteScript should be sufficient for security defense purposes.
        static bool isEventDispatchAllowedInSubtree(Node& node)
        {
#if ASSERT_ENABLED || ENABLE(SECURITY_ASSERTIONS)
            return isScriptAllowed() || EventAllowedScope::isAllowedNode(node);
#else
            UNUSED_PARAM(node);
            return true;
#endif
        }

        static bool hasDisallowedScope()
        {
            ASSERT(isMainThread());
            return s_count;
        }

        static bool isScriptAllowed()
        {
            ASSERT(isMainThread());
#if PLATFORM(IOS_FAMILY)
            return !s_count || !isInWebProcess() || webThreadDelegateMessageScopeCount;
#else
            return !s_count || !isInWebProcess();
#endif
        }
    };

    class InMainThreadOfWebProcess {
    public:
        InMainThreadOfWebProcess()
            : m_isInWebProcess(isInWebProcess())
        {
            ASSERT(isMainThread());
            if (!m_isInWebProcess)
                return;
            ++s_count;
        }

        ~InMainThreadOfWebProcess()
        {
            ASSERT(isMainThread());
            if (!m_isInWebProcess)
                return;
            ASSERT(s_count);
            --s_count;
        }

    private:
        bool m_isInWebProcess;
    };

#if ASSERT_ENABLED
    class EventAllowedScope {
    public:
        explicit EventAllowedScope(ContainerNode& userAgentContentRoot)
            : m_eventAllowedTreeRoot(userAgentContentRoot)
            , m_previousScope(s_currentScope)
        {
            s_currentScope = this;
        }

        ~EventAllowedScope()
        {
            s_currentScope = m_previousScope;
        }

        static bool isAllowedNode(Node& node)
        {
            return s_currentScope && s_currentScope->isAllowedNodeInternal(node);
        }

    private:
        bool isAllowedNodeInternal(Node& node)
        {
            return m_eventAllowedTreeRoot->contains(&node) || (m_previousScope && m_previousScope->isAllowedNodeInternal(node));
        }

        Ref<ContainerNode> m_eventAllowedTreeRoot;

        EventAllowedScope* m_previousScope;
        static EventAllowedScope* s_currentScope;
    };
#else // not ASSERT_ENABLED
    class EventAllowedScope {
    public:
        explicit EventAllowedScope(ContainerNode&) { }
        static bool isAllowedNode(Node&) { return true; }
    };
#endif // not ASSERT_ENABLED

    // FIXME: Remove this class once the sync layout inside SVGImage::draw is removed,
    // CachedSVGFont::ensureCustomFontData no longer synchronously creates a document during style resolution,
    // and refactored the code in RenderFrameBase::performLayoutWithFlattening.
    class DisableAssertionsInScope {
    public:
        DisableAssertionsInScope()
        {
            ASSERT(isMainThread());
            std::swap(s_count, m_originalCount);
        }

        ~DisableAssertionsInScope()
        {
            s_count = m_originalCount;
        }
    private:
        unsigned m_originalCount { 0 };
    };

private:
    WEBCORE_EXPORT static unsigned s_count;
};

} // namespace WebCore
