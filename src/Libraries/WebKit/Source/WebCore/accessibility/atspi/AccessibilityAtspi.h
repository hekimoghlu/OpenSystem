/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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

#if USE(ATSPI)
#include <wtf/CompletionHandler.h>
#include <wtf/FastMalloc.h>
#include <wtf/HashMap.h>
#include <wtf/ListHashSet.h>
#include <wtf/RunLoop.h>
#include <wtf/Vector.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/GUniquePtr.h>

typedef struct _GDBusConnection GDBusConnection;
typedef struct _GDBusInterfaceInfo GDBusInterfaceInfo;
typedef struct _GDBusInterfaceVTable GDBusInterfaceVTable;
typedef struct _GDBusProxy GDBusProxy;
typedef struct _GVariant GVariant;

namespace WebCore {
class AccessibilityObjectAtspi;
class AccessibilityRootAtspi;
enum class AccessibilityRole : uint8_t;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(AccessibilityAtspi);
class AccessibilityAtspi {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(AccessibilityAtspi);
    WTF_MAKE_NONCOPYABLE(AccessibilityAtspi);
    friend NeverDestroyed<AccessibilityAtspi>;
public:
    WEBCORE_EXPORT static AccessibilityAtspi& singleton();

    // Do nothing since this is a singleton.
    void ref() const { }
    void deref() const { }

    void connect(const String&, const String&);

    const char* uniqueName() const;
    GVariant* nullReference() const;
    GVariant* applicationReference() const;
    bool hasClients() const { return !m_clients.isEmpty(); }

    void registerRoot(AccessibilityRootAtspi&, Vector<std::pair<GDBusInterfaceInfo*, GDBusInterfaceVTable*>>&&, CompletionHandler<void(const String&)>&&);
    void unregisterRoot(AccessibilityRootAtspi&);
    String registerObject(AccessibilityObjectAtspi&, Vector<std::pair<GDBusInterfaceInfo*, GDBusInterfaceVTable*>>&&);
    void unregisterObject(AccessibilityObjectAtspi&);
    String registerHyperlink(AccessibilityObjectAtspi&, Vector<std::pair<GDBusInterfaceInfo*, GDBusInterfaceVTable*>>&&);

    void parentChanged(AccessibilityObjectAtspi&);
    void parentChanged(AccessibilityRootAtspi&);
    enum class ChildrenChanged { Added, Removed };
    void childrenChanged(AccessibilityObjectAtspi&, AccessibilityObjectAtspi&, ChildrenChanged);
    void childrenChanged(AccessibilityRootAtspi&, AccessibilityObjectAtspi&, ChildrenChanged);

    void stateChanged(AccessibilityObjectAtspi&, const char*, bool);

    void textChanged(AccessibilityObjectAtspi&, const char*, CString&&, unsigned, unsigned);
    void textAttributesChanged(AccessibilityObjectAtspi&);
    void textCaretMoved(AccessibilityObjectAtspi&, unsigned);
    void textSelectionChanged(AccessibilityObjectAtspi&);

    void valueChanged(AccessibilityObjectAtspi&, double);

    void activeDescendantChanged(AccessibilityObjectAtspi&);

    void selectionChanged(AccessibilityObjectAtspi&);

    void loadEvent(AccessibilityObjectAtspi&, CString&&);

    static const char* localizedRoleName(AccessibilityRole);

#if ENABLE(DEVELOPER_MODE)
    using NotificationObserverParameter = std::variant<std::nullptr_t, String, bool, unsigned, Ref<AccessibilityObjectAtspi>>;
    using NotificationObserver = Function<void(AccessibilityObjectAtspi&, const char*, NotificationObserverParameter)>;
    WEBCORE_EXPORT void addNotificationObserver(void*, NotificationObserver&&);
    WEBCORE_EXPORT void removeNotificationObserver(void*);
#endif

private:
    AccessibilityAtspi();

    struct PendingRootRegistration {
        Ref<AccessibilityRootAtspi> root;
        Vector<std::pair<GDBusInterfaceInfo*, GDBusInterfaceVTable*>> interfaces;
        CompletionHandler<void(const String&)> completionHandler;
    };

    void didConnect(GRefPtr<GDBusConnection>&&);
    void didOwnName();
    void initializeRegistry();
    void addEventListener(const char* dbusName, const char* eventName);
    void removeEventListener(const char* dbusName, const char* eventName);
    void addClient(const char* dbusName);
    void removeClient(const char* dbusName);

    void ensureCache();
    void addToCacheIfNeeded(AccessibilityObjectAtspi&);
    void cacheUpdateTimerFired();
    void cacheClearTimerFired();

    bool shouldEmitSignal(const char* interface, const char* name, const char* detail = "");

#if ENABLE(DEVELOPER_MODE)
    void notify(AccessibilityObjectAtspi&, const char*, NotificationObserverParameter) const;
    void notifyActiveDescendantChanged(AccessibilityObjectAtspi&) const;
    void notifyStateChanged(AccessibilityObjectAtspi&, const char*, bool) const;
    void notifySelectionChanged(AccessibilityObjectAtspi&) const;
    void notifyMenuSelectionChanged(AccessibilityObjectAtspi&) const;
    void notifyTextChanged(AccessibilityObjectAtspi&) const;
    void notifyTextCaretMoved(AccessibilityObjectAtspi&, unsigned) const;
    void notifyValueChanged(AccessibilityObjectAtspi&) const;
    void notifyLoadEvent(AccessibilityObjectAtspi&, const CString&) const;
#endif

    static GDBusInterfaceVTable s_cacheFunctions;

    String m_busName;
    bool m_isConnecting { false };
    GRefPtr<GDBusConnection> m_connection;
    GRefPtr<GDBusProxy> m_registry;
    Vector<PendingRootRegistration> m_pendingRootRegistrations;
    UncheckedKeyHashMap<CString, Vector<GUniquePtr<char*>>> m_eventListeners;
    UncheckedKeyHashMap<AccessibilityRootAtspi*, Vector<unsigned, 3>> m_rootObjects;
    UncheckedKeyHashMap<AccessibilityObjectAtspi*, Vector<unsigned, 7>> m_atspiObjects;
    UncheckedKeyHashMap<AccessibilityObjectAtspi*, Vector<unsigned, 1>> m_atspiHyperlinks;
    UncheckedKeyHashMap<CString, unsigned> m_clients;
    unsigned m_cacheID { 0 };
    UncheckedKeyHashMap<String, AccessibilityObjectAtspi*> m_cache;
    ListHashSet<RefPtr<AccessibilityObjectAtspi>> m_cacheUpdateList;
    RunLoop::Timer m_cacheUpdateTimer;
    RunLoop::Timer m_cacheClearTimer;
#if ENABLE(DEVELOPER_MODE)
    UncheckedKeyHashMap<void*, NotificationObserver> m_notificationObservers;
#endif
};

} // namespace WebCore

#endif // USE(ATSPI)
