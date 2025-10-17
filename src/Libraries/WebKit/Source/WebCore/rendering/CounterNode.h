/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 1, 2024.
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

#include <wtf/CheckedPtr.h>
#include <wtf/Forward.h>
#include <wtf/OptionSet.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

// This implements a counter tree that is used for finding parents in counters() lookup,
// and for propagating count changes when nodes are added or removed.

// Parents represent unique counters and their scope, which are created either explicitly
// by "counter-reset" style rules or implicitly by referring to a counter that is not in scope.
// Such nodes are tagged as "reset" nodes, although they are not all due to "counter-reset".

// Not that render tree children are often counter tree siblings due to counter scoping rules.

namespace WebCore {

class RenderCounter;
class RenderElement;

class CounterNode : public RefCounted<CounterNode>, public CanMakeSingleThreadWeakPtr<CounterNode> {
public:
    enum class Type : uint8_t { Increment, Reset, Set };

    static Ref<CounterNode> create(RenderElement&, OptionSet<Type>, int value);
    ~CounterNode();
    bool actsAsReset() const { return hasResetType() || !m_parent; }
    bool hasResetType() const { return m_type.contains(Type::Reset); }
    bool hasSetType() const { return m_type.contains(Type::Set); }
    int value() const { return m_value; }
    int countInParent() const { return m_countInParent; }
    RenderElement& owner() const;
    void addRenderer(RenderCounter&);
    void removeRenderer(RenderCounter&);

    // Invalidates the text in the renderers of this counter, if any.
    void resetRenderers();

    CounterNode* parent() const { return const_cast<CounterNode*>(m_parent.get()); }
    CounterNode* previousSibling() const { return const_cast<CounterNode*>(m_previousSibling.get()); }
    CounterNode* nextSibling() const { return const_cast<CounterNode*>(m_nextSibling.get()); }
    CounterNode* firstChild() const { return const_cast<CounterNode*>(m_firstChild.get()); }
    CounterNode* lastChild() const { return const_cast<CounterNode*>(m_lastChild.get()); }
    CounterNode* lastDescendant() const;
    CounterNode* previousInPreOrder() const;
    CounterNode* nextInPreOrder(const CounterNode* stayWithin = nullptr) const;
    CounterNode* nextInPreOrderAfterChildren(const CounterNode* stayWithin = nullptr) const;

    void insertAfter(CounterNode& newChild, CounterNode* beforeChild, const AtomString& identifier);
    // identifier must match the identifier of this counter.
    void removeChild(CounterNode&);

private:
    CounterNode(RenderElement&, OptionSet<Type>, int value);
    int computeCountInParent() const;
    // Invalidates the text in the renderer of this counter, if any,
    // and in the renderers of all descendants of this counter, if any.
    void resetThisAndDescendantsRenderers();
    void recount();

    OptionSet<Type> m_type { };
    int m_value;
    int m_countInParent { 0 };
    SingleThreadWeakRef<RenderElement> m_owner;
    SingleThreadWeakPtr<RenderCounter> m_rootRenderer;

    SingleThreadWeakPtr<CounterNode> m_parent;
    SingleThreadWeakPtr<CounterNode> m_previousSibling;
    SingleThreadWeakPtr<CounterNode> m_nextSibling;
    SingleThreadWeakPtr<CounterNode> m_firstChild;
    SingleThreadWeakPtr<CounterNode> m_lastChild;
};

} // namespace WebCore

#if ENABLE(TREE_DEBUGGING)
// Outside the WebCore namespace for ease of invocation from the debugger.
void showCounterTree(const WebCore::CounterNode*);
#endif
