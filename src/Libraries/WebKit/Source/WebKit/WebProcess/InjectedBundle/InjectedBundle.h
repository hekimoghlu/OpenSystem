/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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

#include "APIInjectedBundleBundleClient.h"
#include "APIObject.h"
#include "SandboxExtension.h"
#include <JavaScriptCore/JavaScript.h>
#include <WebCore/UserContentTypes.h>
#include <WebCore/UserScriptTypes.h>
#include <wtf/RefPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/UUID.h>
#include <wtf/text/WTFString.h>

#if USE(GLIB)
typedef struct _GModule GModule;
#endif

#if USE(FOUNDATION)
OBJC_CLASS NSSet;
OBJC_CLASS NSBundle;
OBJC_CLASS NSMutableDictionary;
OBJC_CLASS WKWebProcessBundleParameters;
#endif

namespace API {
class Array;
class Data;
}

namespace IPC {
class Decoder;
class Connection;
}

namespace WebKit {

#if USE(FOUNDATION)
typedef NSBundle *PlatformBundle;
#elif USE(GLIB)
typedef ::GModule* PlatformBundle;
#else
typedef void* PlatformBundle;
#endif

class InjectedBundleScriptWorld;
class WebFrame;
class WebPage;
class WebPageGroupProxy;
struct WebProcessCreationParameters;

class InjectedBundle : public API::ObjectImpl<API::Object::Type::Bundle> {
public:
    static RefPtr<InjectedBundle> create(WebProcessCreationParameters&, RefPtr<API::Object>&& initializationUserData);

    ~InjectedBundle();

    bool initialize(const WebProcessCreationParameters&, RefPtr<API::Object>&& initializationUserData);

    void setBundleParameter(const String&, std::span<const uint8_t>);
    void setBundleParameters(std::span<const uint8_t>);

    // API
    void setClient(std::unique_ptr<API::InjectedBundle::Client>&&);
    void postMessage(const String&, API::Object*);
    void postSynchronousMessage(const String&, API::Object*, RefPtr<API::Object>& returnData);
    void setServiceWorkerProxyCreationCallback(void (*)(uint64_t));

    // TestRunner only SPI
    void addOriginAccessAllowListEntry(const String&, const String&, const String&, bool);
    void removeOriginAccessAllowListEntry(const String&, const String&, const String&, bool);
    void resetOriginAccessAllowLists();
    void setAsynchronousSpellCheckingEnabled(bool);
    int numberOfPages(WebFrame*, double, double);
    int pageNumberForElementById(WebFrame*, const String&, double, double);
    String pageSizeAndMarginsInPixels(WebFrame*, int, int, int, int, int, int, int);
    bool isPageBoxVisible(WebFrame*, int);
    void setUserStyleSheetLocation(const String&);
    void removeAllWebNotificationPermissions(WebPage*);
    std::optional<WTF::UUID> webNotificationID(JSContextRef, JSValueRef);
    Ref<API::Data> createWebDataFromUint8Array(JSContextRef, JSValueRef);
    
    typedef HashMap<WTF::UUID, String> DocumentIDToURLMap;
    DocumentIDToURLMap liveDocumentURLs(bool excludeDocumentsInPageGroupPages);

    // Garbage collection API
    void garbageCollectJavaScriptObjects();
    void garbageCollectJavaScriptObjectsOnAlternateThreadForDebugging(bool waitUntilDone);
    size_t javaScriptObjectsCount();

    // Callback hooks
    void didCreatePage(WebPage&);
    void willDestroyPage(WebPage&);
    void didReceiveMessage(const String&, RefPtr<API::Object>&&);
    void didReceiveMessageToPage(WebPage&, const String&, RefPtr<API::Object>&&);

    static void reportException(JSContextRef, JSValueRef exception);

    static bool isProcessingUserGesture();

    void setTabKeyCyclesThroughElements(WebPage*, bool enabled);
    void setSerialLoadingEnabled(bool);
    void setAccessibilityIsolatedTreeEnabled(bool);
    void dispatchPendingLoadRequests();

#if PLATFORM(COCOA)
    WKWebProcessBundleParameters *bundleParameters();

    void extendClassesForParameterCoder(API::Array& classes);
    NSSet* classesForCoder();
#endif

private:
    explicit InjectedBundle(const WebProcessCreationParameters&);

#if PLATFORM(COCOA)
    bool decodeBundleParameters(API::Data*);
#endif

    String m_path;
    PlatformBundle m_platformBundle; // This is leaked right now, since we never unload the bundle/module.

    RefPtr<SandboxExtension> m_sandboxExtension;

    std::unique_ptr<API::InjectedBundle::Client> m_client;

#if PLATFORM(COCOA)
    RetainPtr<WKWebProcessBundleParameters> m_bundleParameters;
    RetainPtr<NSSet> m_classesForCoder;
#endif
};

} // namespace WebKit
