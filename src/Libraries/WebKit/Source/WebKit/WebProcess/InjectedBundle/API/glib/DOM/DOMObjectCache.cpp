/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 16, 2021.
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
#include "config.h"
#include "DOMObjectCache.h"

#include <WebCore/Document.h>
#include <WebCore/FrameDestructionObserver.h>
#include <WebCore/FrameDestructionObserverInlines.h>
#include <WebCore/LocalDOMWindow.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/Node.h>
#include <glib-object.h>
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>
#include <wtf/glib/GRefPtr.h>

namespace WebKit {

struct DOMObjectCacheData {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    DOMObjectCacheData(GObject* wrapper)
        : object(wrapper)
        , cacheReferences(1)
    {
    }

    void clearObject()
    {
        ASSERT(object);
        ASSERT(cacheReferences >= 1);
        ASSERT(object->ref_count >= 1);

        // Make sure we don't unref more than the references the object actually has. It can happen that user
        // unreffed a reference owned by the cache.
        cacheReferences = std::min(static_cast<unsigned>(object->ref_count), cacheReferences);
        GRefPtr<GObject> protect(object);
        do {
            g_object_unref(object);
        } while (--cacheReferences);
        object = nullptr;
    }

    void* refObject()
    {
        ASSERT(object);

        cacheReferences++;
        return g_object_ref(object);
    }

    GObject* object;
    unsigned cacheReferences;
};

class DOMObjectCacheFrameObserver;
typedef HashMap<WebCore::LocalFrame*, std::unique_ptr<DOMObjectCacheFrameObserver>> DOMObjectCacheFrameObserverMap;

static DOMObjectCacheFrameObserverMap& domObjectCacheFrameObservers()
{
    static NeverDestroyed<DOMObjectCacheFrameObserverMap> map;
    return map;
}

class DOMObjectCacheFrameObserver final: public WebCore::FrameDestructionObserver {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(DOMObjectCacheFrameObserver);
public:
    DOMObjectCacheFrameObserver(WebCore::LocalFrame& frame)
        : FrameDestructionObserver(&frame)
    {
    }

    ~DOMObjectCacheFrameObserver()
    {
        ASSERT(m_objects.isEmpty());
    }

    void addObjectCacheData(DOMObjectCacheData& data)
    {
        ASSERT(!m_objects.contains(&data));

        auto* domWindow = m_frame->document()->domWindow();
        if (domWindow && (!m_domWindowObserver || m_domWindowObserver->window() != domWindow)) {
            // New LocalDOMWindow, clear the cache and create a new DOMWindowObserver.
            clear();
            m_domWindowObserver = makeUnique<DOMWindowObserver>(*domWindow, *this);
        }

        m_objects.append(&data);
        g_object_weak_ref(data.object, DOMObjectCacheFrameObserver::objectFinalizedCallback, this);
    }

private:
    class DOMWindowObserver final : public WebCore::LocalDOMWindowObserver {
        WTF_MAKE_TZONE_ALLOCATED_INLINE(DOMWindowObserver);
    public:
        DOMWindowObserver(WebCore::LocalDOMWindow& window, DOMObjectCacheFrameObserver& frameObserver)
            : m_window(window)
            , m_frameObserver(frameObserver)
        {
            window.registerObserver(*this);
        }

        ~DOMWindowObserver()
        {
            if (m_window)
                m_window->unregisterObserver(*this);
        }

        WebCore::LocalDOMWindow* window() const { return m_window.get(); }

    private:
        void willDetachGlobalObjectFromFrame() override
        {
            m_frameObserver.willDetachGlobalObjectFromFrame();
        }

        WeakPtr<WebCore::LocalDOMWindow, WebCore::WeakPtrImplWithEventTargetData> m_window;
        DOMObjectCacheFrameObserver& m_frameObserver;
    };

    static void objectFinalizedCallback(gpointer userData, GObject* finalizedObject)
    {
        DOMObjectCacheFrameObserver* observer = static_cast<DOMObjectCacheFrameObserver*>(userData);
        observer->m_objects.removeFirstMatching([finalizedObject](DOMObjectCacheData* data) {
            return data->object == finalizedObject;
        });
    }

    void clear()
    {
        if (m_objects.isEmpty())
            return;

        auto objects = WTFMove(m_objects);

        // Deleting of DOM wrappers might end up deleting the wrapped core object which could cause some problems
        // for example if a Document is deleted during the frame destruction, so we remove the weak references now
        // and delete the objects on next run loop iteration. See https://bugs.webkit.org/show_bug.cgi?id=151700.
        for (auto* data : objects)
            g_object_weak_unref(data->object, DOMObjectCacheFrameObserver::objectFinalizedCallback, this);

        RunLoop::main().dispatch([objects] {
            for (auto* data : objects)
                data->clearObject();
        });
    }

    void willDetachPage() override
    {
        clear();
    }

    void frameDestroyed() override
    {
        clear();
        auto* frame = m_frame.get();
        FrameDestructionObserver::frameDestroyed();
        domObjectCacheFrameObservers().remove(frame);
    }

    void willDetachGlobalObjectFromFrame()
    {
        clear();
        m_domWindowObserver = nullptr;
    }

    Vector<DOMObjectCacheData*, 8> m_objects;
    std::unique_ptr<DOMWindowObserver> m_domWindowObserver;
};

static DOMObjectCacheFrameObserver& getOrCreateDOMObjectCacheFrameObserver(WebCore::LocalFrame& frame)
{
    DOMObjectCacheFrameObserverMap::AddResult result = domObjectCacheFrameObservers().add(&frame, nullptr);
    if (result.isNewEntry)
        result.iterator->value = makeUnique<DOMObjectCacheFrameObserver>(frame);
    return *result.iterator->value;
}

typedef HashMap<void*, std::unique_ptr<DOMObjectCacheData>> DOMObjectMap;

static DOMObjectMap& domObjects()
{
    static NeverDestroyed<DOMObjectMap> staticDOMObjects;
    return staticDOMObjects;
}

void DOMObjectCache::forget(void* objectHandle)
{
    ASSERT(domObjects().contains(objectHandle));
    domObjects().remove(objectHandle);
}

void* DOMObjectCache::get(void* objectHandle)
{
    DOMObjectCacheData* data = domObjects().get(objectHandle);
    return data ? data->refObject() : nullptr;
}

void DOMObjectCache::put(void* objectHandle, void* wrapper)
{
    DOMObjectMap::AddResult result = domObjects().add(objectHandle, nullptr);
    if (result.isNewEntry)
        result.iterator->value = makeUnique<DOMObjectCacheData>(G_OBJECT(wrapper));
}

void DOMObjectCache::put(WebCore::Node* objectHandle, void* wrapper)
{
    DOMObjectMap::AddResult result = domObjects().add(objectHandle, nullptr);
    if (!result.isNewEntry)
        return;

    result.iterator->value = makeUnique<DOMObjectCacheData>(G_OBJECT(wrapper));
    if (auto* frame = objectHandle->document().frame())
        getOrCreateDOMObjectCacheFrameObserver(*frame).addObjectCacheData(*result.iterator->value);
}

}
