/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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

#include "ContextMenu.h"
#include "ContextMenuProvider.h"
#include "ExceptionOr.h"
#include "InspectorFrontendClient.h"
#include <JavaScriptCore/JSCJSValue.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CanvasPath;
class CanvasRenderingContext2D;
class DOMWrapperWorld;
class DeferredPromise;
class Event;
class File;
class FrontendMenuProvider;
class HTMLIFrameElement;
class OffscreenCanvasRenderingContext2D;
class Page;
class Path2D;

class InspectorFrontendHost : public RefCounted<InspectorFrontendHost> {
public:
    static Ref<InspectorFrontendHost> create(InspectorFrontendClient* client, Page* frontendPage)
    {
        return adoptRef(*new InspectorFrontendHost(client, frontendPage));
    }

    WEBCORE_EXPORT ~InspectorFrontendHost();
    WEBCORE_EXPORT void disconnectClient();

    WEBCORE_EXPORT void addSelfToGlobalObjectInWorld(DOMWrapperWorld&);

    void loaded();
    void closeWindow();
    void reopen();
    void reset();

    void bringToFront();
    void inspectedURLChanged(const String&);

    bool supportsShowCertificate() const;
    bool showCertificate(const String& serializedCertificate);

    void setZoomFactor(float);
    float zoomFactor();

    void setForcedAppearance(String);

    String userInterfaceLayoutDirection();

    bool supportsDockSide(const String&);
    void requestSetDockSide(const String&);

    void setAttachedWindowHeight(unsigned);
    void setAttachedWindowWidth(unsigned);

    void setSheetRect(float x, float y, unsigned width, unsigned height);

    void startWindowDrag();
    void moveWindowBy(float x, float y) const;

    bool isRemote() const;
    String localizedStringsURL() const;
    String backendCommandsURL() const;
    unsigned inspectionLevel() const;

    String platform() const;
    String platformVersionName() const;

    struct DebuggableInfo {
        String debuggableType;
        String targetPlatformName;
        String targetBuildVersion;
        String targetProductVersion;
        bool targetIsSimulator;
    };
    DebuggableInfo debuggableInfo() const;

    void copyText(const String& text);
    void killText(const String& text, bool shouldPrependToKillRing, bool shouldStartNewSequence);

    void openURLExternally(const String& url);
    void revealFileExternally(const String& path);

    using SaveMode = InspectorFrontendClient::SaveMode;
    using SaveData = InspectorFrontendClient::SaveData;
    bool canSave(SaveMode);
    void save(Vector<SaveData>&&, bool forceSaveAs);

    bool canLoad();
    void load(const String& path, Ref<DeferredPromise>&&);

    bool canPickColorFromScreen();
    void pickColorFromScreen(Ref<DeferredPromise>&&);

    struct ContextMenuItem {
        String type;
        String label;
        std::optional<int> id;
        std::optional<bool> enabled;
        std::optional<bool> checked;
        std::optional<Vector<ContextMenuItem>> subItems;
    };
    void showContextMenu(Event&, Vector<ContextMenuItem>&&);

    void sendMessageToBackend(const String& message);
    void dispatchEventAsContextMenuEvent(Event&);

    bool isUnderTest();
    void unbufferedLog(const String& message);

    void beep();
    void inspectInspector();
    bool isBeingInspected();
    void setAllowsInspectingInspector(bool);

    bool engineeringSettingsAllowed();

    bool supportsDiagnosticLogging();
#if ENABLE(INSPECTOR_TELEMETRY)
    bool diagnosticLoggingAvailable();
    void logDiagnosticEvent(const String& eventName, const String& payload);
#endif

    bool supportsWebExtensions();
#if ENABLE(INSPECTOR_EXTENSIONS)
    void didShowExtensionTab(const String& extensionID, const String& extensionTabID, HTMLIFrameElement& extensionFrame);
    void didHideExtensionTab(const String& extensionID, const String& extensionTabID);
    void didNavigateExtensionTab(const String& extensionID, const String& extensionTabID, const String& url);
    void inspectedPageDidNavigate(const String& url);
    ExceptionOr<JSC::JSValue> evaluateScriptInExtensionTab(HTMLIFrameElement& extensionFrame, const String& scriptSource);
#endif

    // IDL extensions.

    String getPath(const File&) const;

    float getCurrentX(const CanvasRenderingContext2D&) const;
    float getCurrentY(const CanvasRenderingContext2D&) const;
    Ref<Path2D> getPath(const CanvasRenderingContext2D&) const;
    void setPath(CanvasRenderingContext2D&, Path2D&) const;

#if ENABLE(OFFSCREEN_CANVAS)
    float getCurrentX(const OffscreenCanvasRenderingContext2D&) const;
    float getCurrentY(const OffscreenCanvasRenderingContext2D&) const;
    Ref<Path2D> getPath(const OffscreenCanvasRenderingContext2D&) const;
    void setPath(OffscreenCanvasRenderingContext2D&, Path2D&) const;
#endif

private:
#if ENABLE(CONTEXT_MENUS)
    friend class FrontendMenuProvider;
#endif
    WEBCORE_EXPORT InspectorFrontendHost(InspectorFrontendClient*, Page* frontendPage);

    InspectorFrontendClient* m_client;
    WeakPtr<Page> m_frontendPage;
#if ENABLE(CONTEXT_MENUS)
    FrontendMenuProvider* m_menuProvider;
#endif
};

} // namespace WebCore
