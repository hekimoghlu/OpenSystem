/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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

#include "CustomElementFormValue.h"
#include "Element.h"
#include "GCReachableRef.h"
#include "QualifiedName.h"
#include <wtf/CheckedRef.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/AtomString.h>

namespace JSC {

class JSGlobalObject;
class CallFrame;

}

namespace WebCore {

class CustomElementQueue;
class Document;
class Element;
class HTMLFormElement;
class JSCustomElementInterface;

class CustomElementReactionQueueItem {
    WTF_MAKE_TZONE_ALLOCATED(CustomElementReactionQueueItem);
    WTF_MAKE_NONCOPYABLE(CustomElementReactionQueueItem);
public:
    enum class Type : uint8_t {
        Invalid,
        ElementUpgrade,
        Connected,
        Disconnected,
        Adopted,
        AttributeChanged,
        FormAssociated,
        FormReset,
        FormDisabled,
        FormStateRestore,
    };

    struct AdoptedPayload {
        Ref<Document> oldDocument;
        Ref<Document> newDocument;
        ~AdoptedPayload();
    };

    struct FormAssociatedPayload {
        RefPtr<HTMLFormElement> form;
        ~FormAssociatedPayload();
    };

    using AttributeChangedPayload = std::tuple<QualifiedName, AtomString, AtomString>;
    using FormDisabledPayload = bool;
    using FormStateRestorePayload = CustomElementFormValue;
    using Payload = std::optional<std::variant<AdoptedPayload, AttributeChangedPayload, FormAssociatedPayload, FormDisabledPayload, FormStateRestorePayload>>;

    CustomElementReactionQueueItem();
    CustomElementReactionQueueItem(CustomElementReactionQueueItem&&);
    CustomElementReactionQueueItem(Type, Payload = std::nullopt);
    ~CustomElementReactionQueueItem();
    Type type() const { return m_type; }
    void invoke(Element&, JSCustomElementInterface&);

private:
    Type m_type { Type::Invalid };
    Payload m_payload;
};

// https://html.spec.whatwg.org/multipage/custom-elements.html#element-queue
class CustomElementQueue {
    WTF_MAKE_TZONE_ALLOCATED(CustomElementQueue);
    WTF_MAKE_NONCOPYABLE(CustomElementQueue);
public:
    CustomElementQueue() = default;
    ~CustomElementQueue() { ASSERT(isEmpty()); }

    bool isEmpty() const { return m_elements.isEmpty(); }
    void add(Element&);
    WEBCORE_EXPORT void processQueue(JSC::JSGlobalObject*);

    Vector<Ref<Element>, 4> takeElements();

private:
    void invokeAll();

    Vector<Ref<Element>, 4> m_elements;
    bool m_invoking { false };
};

class CustomElementReactionQueue final : public CanMakeCheckedPtr<CustomElementReactionQueue> {
    WTF_MAKE_TZONE_ALLOCATED(CustomElementReactionQueue);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(CustomElementReactionQueue);
    WTF_MAKE_NONCOPYABLE(CustomElementReactionQueue);
public:
    CustomElementReactionQueue(JSCustomElementInterface&);
    ~CustomElementReactionQueue();

    static void enqueueElementUpgrade(Element&, bool alreadyScheduledToUpgrade);
    static void tryToUpgradeElement(Element&);
    static void enqueueConnectedCallbackIfNeeded(Element&);
    static void enqueueDisconnectedCallbackIfNeeded(Element&);
    static void enqueueAdoptedCallbackIfNeeded(Element&, Document& oldDocument, Document& newDocument);
    static void enqueueAttributeChangedCallbackIfNeeded(Element&, const QualifiedName&, const AtomString& oldValue, const AtomString& newValue);
    static void enqueueFormAssociatedCallbackIfNeeded(Element&, HTMLFormElement*);
    static void enqueueFormDisabledCallbackIfNeeded(Element&, bool isDisabled);
    static void enqueueFormResetCallbackIfNeeded(Element&);
    static void enqueueFormStateRestoreCallbackIfNeeded(Element&, CustomElementFormValue&&);
    static void enqueuePostUpgradeReactions(Element&);

    bool observesStyleAttribute() const;
    bool isElementInternalsDisabled() const;
    bool isElementInternalsAttached() const;
    void setElementInternalsAttached();
    bool isFormAssociated() const;
    bool hasFormStateRestoreCallback() const;

    void invokeAll(Element&);
    void clear();
    bool isEmpty() const { return m_items.isEmpty(); }
#if ASSERT_ENABLED
    bool hasJustUpgradeReaction() const;
#endif

    static void processBackupQueue(CustomElementQueue&);

    static void enqueueElementsOnAppropriateElementQueue(const Vector<Ref<Element>>&);

private:
    static void enqueueElementOnAppropriateElementQueue(Element&);

    using Item = CustomElementReactionQueueItem;

    Ref<JSCustomElementInterface> m_interface;
    Vector<Item, 1> m_items;
    bool m_elementInternalsAttached { false };
};

class CustomElementReactionDisallowedScope {
public:
    CustomElementReactionDisallowedScope()
    {
#if ASSERT_ENABLED
        s_customElementReactionDisallowedCount++;
#endif
    }

    ~CustomElementReactionDisallowedScope()
    {
#if ASSERT_ENABLED
        ASSERT(s_customElementReactionDisallowedCount);
        s_customElementReactionDisallowedCount--;
#endif
    }

#if ASSERT_ENABLED
    static bool isReactionAllowed() { return !s_customElementReactionDisallowedCount; }
#endif

    class AllowedScope {
#if ASSERT_ENABLED
    public:
        AllowedScope()
            : m_originalCount(s_customElementReactionDisallowedCount)
        {
            s_customElementReactionDisallowedCount = 0;
        }

        ~AllowedScope()
        {
            s_customElementReactionDisallowedCount = m_originalCount;
        }

    private:
        unsigned m_originalCount;
#endif // ASSERT_ENABLED
    };

private:
#if ASSERT_ENABLED
    WEBCORE_EXPORT static unsigned s_customElementReactionDisallowedCount;

    friend class AllowedScope;
#endif
};

class CustomElementReactionStack : public CustomElementReactionDisallowedScope::AllowedScope {
public:
    ALWAYS_INLINE CustomElementReactionStack(JSC::JSGlobalObject* state)
        : m_previousProcessingStack(s_currentProcessingStack)
        , m_state(state)
    {
        s_currentProcessingStack = this;
    }

    ALWAYS_INLINE CustomElementReactionStack(JSC::JSGlobalObject& state)
        : CustomElementReactionStack(&state)
    { }

    ALWAYS_INLINE ~CustomElementReactionStack()
    {
        if (UNLIKELY(!m_queue.isEmpty()))
            m_queue.processQueue(m_state);
        s_currentProcessingStack = m_previousProcessingStack;
    }

    Vector<Ref<Element>, 4> takeElements() { return m_queue.takeElements(); }

private:
    CustomElementQueue m_queue;
    CustomElementReactionStack* const m_previousProcessingStack;
    JSC::JSGlobalObject* const m_state;

    WEBCORE_EXPORT static CustomElementReactionStack* s_currentProcessingStack;

    friend CustomElementReactionQueue;
};

}
