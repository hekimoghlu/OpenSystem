/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 28, 2024.
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

#include "ShadowRoot.h"
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakHashMap.h>
#include <wtf/WeakHashSet.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/AtomString.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

class Element;
class HTMLSlotElement;
class Node;

class SlotAssignment {
    WTF_MAKE_TZONE_ALLOCATED(SlotAssignment);
    WTF_MAKE_NONCOPYABLE(SlotAssignment);
public:
    SlotAssignment() = default;
    virtual ~SlotAssignment() = default;

    // These functions are only useful for NamedSlotAssignment but it's here to avoid virtual function calls in perf critical code paths.
    void resolveSlotsBeforeNodeInsertionOrRemoval();
    void willRemoveAllChildren();

    virtual HTMLSlotElement* findAssignedSlot(const Node&) = 0;
    virtual const Vector<WeakPtr<Node, WeakPtrImplWithEventTargetData>>* assignedNodesForSlot(const HTMLSlotElement&, ShadowRoot&) = 0;

    virtual void renameSlotElement(HTMLSlotElement&, const AtomString& oldName, const AtomString& newName, ShadowRoot&) = 0;
    virtual void addSlotElementByName(const AtomString&, HTMLSlotElement&, ShadowRoot&) = 0;
    virtual void removeSlotElementByName(const AtomString&, HTMLSlotElement&, ContainerNode* oldParentOfRemovedTreeForRemoval, ShadowRoot&) = 0;
    virtual void slotManualAssignmentDidChange(HTMLSlotElement&, Vector<WeakPtr<Node, WeakPtrImplWithEventTargetData>>& previous, Vector<WeakPtr<Node, WeakPtrImplWithEventTargetData>>& current, ShadowRoot&) = 0;
    virtual void didRemoveManuallyAssignedNode(HTMLSlotElement&, const Node&, ShadowRoot&) = 0;
    virtual void slotFallbackDidChange(HTMLSlotElement&, ShadowRoot&) = 0;

    virtual void hostChildElementDidChange(const Element&, ShadowRoot&) = 0;
    virtual void hostChildElementDidChangeSlotAttribute(Element&, const AtomString& oldValue, const AtomString& newValue, ShadowRoot&) = 0;

    virtual void willRemoveAssignedNode(const Node&, ShadowRoot&) = 0;
    virtual void didRemoveAllChildrenOfShadowHost(ShadowRoot&) = 0;
    virtual void didMutateTextNodesOfShadowHost(ShadowRoot&) = 0;

protected:
    // These flags are used by NamedSlotAssignment but it's here to avoid virtual function calls in perf critical code paths.
    bool m_slotAssignmentsIsValid { false };
    bool m_willBeRemovingAllChildren { false };
    unsigned m_slotMutationVersion { 0 };
};

class NamedSlotAssignment : public SlotAssignment {
    WTF_MAKE_TZONE_ALLOCATED(NamedSlotAssignment);
    WTF_MAKE_NONCOPYABLE(NamedSlotAssignment);
public:
    NamedSlotAssignment();
    virtual ~NamedSlotAssignment();

    static const AtomString& defaultSlotName() { return emptyAtom(); }

protected:
    void didChangeSlot(const AtomString&, ShadowRoot&);

private:
    HTMLSlotElement* findAssignedSlot(const Node&) final;

    void renameSlotElement(HTMLSlotElement&, const AtomString& oldName, const AtomString& newName, ShadowRoot&) final;
    void addSlotElementByName(const AtomString&, HTMLSlotElement&, ShadowRoot&) final;
    void removeSlotElementByName(const AtomString&, HTMLSlotElement&, ContainerNode* oldParentOfRemovedTreeForRemoval, ShadowRoot&) final;
    void slotManualAssignmentDidChange(HTMLSlotElement&, Vector<WeakPtr<Node, WeakPtrImplWithEventTargetData>>& previous, Vector<WeakPtr<Node, WeakPtrImplWithEventTargetData>>& current, ShadowRoot&) final;
    void didRemoveManuallyAssignedNode(HTMLSlotElement&, const Node&, ShadowRoot&) final;
    void slotFallbackDidChange(HTMLSlotElement&, ShadowRoot&) final;

    const Vector<WeakPtr<Node, WeakPtrImplWithEventTargetData>>* assignedNodesForSlot(const HTMLSlotElement&, ShadowRoot&) final;
    void willRemoveAssignedNode(const Node&, ShadowRoot&) final;

    void didRemoveAllChildrenOfShadowHost(ShadowRoot&) final;
    void didMutateTextNodesOfShadowHost(ShadowRoot&) final;
    void hostChildElementDidChange(const Element&, ShadowRoot&) override;
    void hostChildElementDidChangeSlotAttribute(Element&, const AtomString& oldValue, const AtomString& newValue, ShadowRoot&) final;

    struct Slot {
        WTF_MAKE_TZONE_ALLOCATED(Slot);
    public:
        bool hasSlotElements() { return !!elementCount; }
        bool hasDuplicatedSlotElements() { return elementCount > 1; }
        bool shouldResolveSlotElement() { return !element && elementCount; }

        WeakPtr<HTMLSlotElement, WeakPtrImplWithEventTargetData> element;
        WeakPtr<HTMLSlotElement, WeakPtrImplWithEventTargetData> oldElement; // Set by resolveSlotsAfterSlotMutation to dispatch slotchange in tree order.
        unsigned elementCount { 0 };
        bool seenFirstElement { false }; // Used in resolveSlotsAfterSlotMutation.
        Vector<WeakPtr<Node, WeakPtrImplWithEventTargetData>> assignedNodes;
    };

    bool hasAssignedNodes(ShadowRoot&, Slot&);
    enum class SlotMutationType { Insertion, Removal };
    void resolveSlotsAfterSlotMutation(ShadowRoot&, SlotMutationType, ContainerNode* oldParentOfRemovedTree = nullptr);

    virtual const AtomString& slotNameForHostChild(const Node&) const;

    HTMLSlotElement* findFirstSlotElement(Slot&);

    void assignSlots(ShadowRoot&);
    void assignToSlot(Node& child, const AtomString& slotName);

    unsigned m_slotResolutionVersion { 0 };
    unsigned m_slotElementCount { 0 };

    UncheckedKeyHashMap<AtomString, std::unique_ptr<Slot>> m_slots;

#if ASSERT_ENABLED
    WeakHashSet<HTMLSlotElement, WeakPtrImplWithEventTargetData> m_slotElementsForConsistencyCheck;
#endif
};

class ManualSlotAssignment : public SlotAssignment {
public:
    ManualSlotAssignment() = default;

    HTMLSlotElement* findAssignedSlot(const Node&) final;

    const Vector<WeakPtr<Node, WeakPtrImplWithEventTargetData>>* assignedNodesForSlot(const HTMLSlotElement&, ShadowRoot&) final;
    void renameSlotElement(HTMLSlotElement&, const AtomString&, const AtomString&, ShadowRoot&) final;
    void addSlotElementByName(const AtomString&, HTMLSlotElement&, ShadowRoot&) final;
    void removeSlotElementByName(const AtomString&, HTMLSlotElement&, ContainerNode*, ShadowRoot&) final;
    void slotManualAssignmentDidChange(HTMLSlotElement&, Vector<WeakPtr<Node, WeakPtrImplWithEventTargetData>>& previous, Vector<WeakPtr<Node, WeakPtrImplWithEventTargetData>>& current, ShadowRoot&) final;
    void didRemoveManuallyAssignedNode(HTMLSlotElement&, const Node&, ShadowRoot&) final;
    void slotFallbackDidChange(HTMLSlotElement&, ShadowRoot&) final;

    void hostChildElementDidChange(const Element&, ShadowRoot&) final;
    void hostChildElementDidChangeSlotAttribute(Element&, const AtomString&, const AtomString&, ShadowRoot&) final;

    void willRemoveAssignedNode(const Node&, ShadowRoot&) final;
    void didRemoveAllChildrenOfShadowHost(ShadowRoot&) final;
    void didMutateTextNodesOfShadowHost(ShadowRoot&) final;

private:
    struct Slot {
        Vector<WeakPtr<Node, WeakPtrImplWithEventTargetData>> cachedAssignment;
        uint64_t cachedVersion { 0 };
    };
    WeakHashMap<HTMLSlotElement, Slot, WeakPtrImplWithEventTargetData> m_slots;
    uint64_t m_slottableVersion { 0 };
    unsigned m_slotElementCount { 0 };
};

inline void SlotAssignment::resolveSlotsBeforeNodeInsertionOrRemoval()
{
    m_slotMutationVersion++;
    m_willBeRemovingAllChildren = false;
}

inline void SlotAssignment::willRemoveAllChildren()
{
    m_slotMutationVersion++;
    m_willBeRemovingAllChildren = true;
}

inline void ShadowRoot::resolveSlotsBeforeNodeInsertionOrRemoval()
{
    if (UNLIKELY(m_slotAssignment))
        m_slotAssignment->resolveSlotsBeforeNodeInsertionOrRemoval();
}

inline void ShadowRoot::willRemoveAllChildren(ContainerNode&)
{
    if (UNLIKELY(m_slotAssignment))
        m_slotAssignment->willRemoveAllChildren();
}

inline void ShadowRoot::didRemoveAllChildrenOfShadowHost()
{
    if (UNLIKELY(m_slotAssignment))
        m_slotAssignment->didRemoveAllChildrenOfShadowHost(*this);
}

inline void ShadowRoot::didMutateTextNodesOfShadowHost()
{
    if (UNLIKELY(m_slotAssignment))
        m_slotAssignment->didMutateTextNodesOfShadowHost(*this);
}

inline void ShadowRoot::hostChildElementDidChange(const Element& childElement)
{
    if (UNLIKELY(m_slotAssignment))
        m_slotAssignment->hostChildElementDidChange(childElement, *this);
}

inline void ShadowRoot::hostChildElementDidChangeSlotAttribute(Element& element, const AtomString& oldValue, const AtomString& newValue)
{
    if (!m_slotAssignment)
        return;
    m_slotAssignment->hostChildElementDidChangeSlotAttribute(element, oldValue, newValue, *this);
}

inline void ShadowRoot::willRemoveAssignedNode(const Node& node)
{
    if (m_slotAssignment)
        m_slotAssignment->willRemoveAssignedNode(node, *this);
}

} // namespace WebCore
