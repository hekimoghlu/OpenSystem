/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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

#include "LocalDOMWindowProperty.h"
#include "NavigatorBase.h"
#include "ScriptWrappable.h"
#include "ShareData.h"
#include "Supplementable.h"
#include <wtf/CheckedPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class Blob;
class DeferredPromise;
class DOMMimeTypeArray;
class DOMPluginArray;
class Page;
class ShareDataReader;

class Navigator final
    : public NavigatorBase
    , public ScriptWrappable
    , public LocalDOMWindowProperty
    , public Supplementable<Navigator>
{
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(Navigator);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(Navigator);
public:
    static Ref<Navigator> create(ScriptExecutionContext* context, LocalDOMWindow& window) { return adoptRef(*new Navigator(context, window)); }
    virtual ~Navigator();

    String appVersion() const;
    DOMPluginArray& plugins();
    DOMMimeTypeArray& mimeTypes();
    bool pdfViewerEnabled();
    bool cookieEnabled() const;
    bool javaEnabled() const { return false; }
    const String& userAgent() const final;
    String platform() const final;
    void userAgentChanged();
    bool onLine() const final;
    bool canShare(Document&, const ShareData&);
    void share(Document&, const ShareData&, Ref<DeferredPromise>&&);
    
#if ENABLE(NAVIGATOR_STANDALONE)
    bool standalone() const;
#endif

    int maxTouchPoints() const;

    GPU* gpu();

    Page* page();
    RefPtr<Page> protectedPage();

    const Document* document() const;
    Document* document();
    RefPtr<Document> protectedDocument();

    void setAppBadge(std::optional<unsigned long long>, Ref<DeferredPromise>&&);
    void clearAppBadge(Ref<DeferredPromise>&&);

    void setClientBadge(std::optional<unsigned long long>, Ref<DeferredPromise>&&);
    void clearClientBadge(Ref<DeferredPromise>&&);

private:
    void showShareData(ExceptionOr<ShareDataWithParsedURL&>, Ref<DeferredPromise>&&);
    explicit Navigator(ScriptExecutionContext*, LocalDOMWindow&);

    void initializePluginAndMimeTypeArrays();

    mutable RefPtr<ShareDataReader> m_loader;
    mutable bool m_hasPendingShare { false };
    mutable bool m_pdfViewerEnabled { false };
    mutable RefPtr<DOMPluginArray> m_plugins;
    mutable RefPtr<DOMMimeTypeArray> m_mimeTypes;
    mutable String m_userAgent;
    mutable String m_platform;
    RefPtr<GPU> m_gpuForWebGPU;
};
}
