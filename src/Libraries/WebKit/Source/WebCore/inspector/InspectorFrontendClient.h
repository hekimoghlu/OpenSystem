/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 14, 2022.
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

#include "CertificateInfo.h"
#include "Color.h"
#include "DiagnosticLoggingClient.h"
#include "FrameIdentifier.h"
#include "InspectorDebuggableType.h"
#include "UserInterfaceLayoutDirection.h"
#include <wtf/Forward.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

#if ENABLE(INSPECTOR_EXTENSIONS)
namespace Inspector {
using ExtensionID = String;
using ExtensionTabID = String;
}
#endif

namespace WebCore {
class InspectorFrontendClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::InspectorFrontendClient> : std::true_type { };
}

namespace WebCore {

class FloatRect;
class InspectorFrontendAPIDispatcher;
class Page;

enum class InspectorFrontendClientAppearance : uint8_t {
    System,
    Light,
    Dark,
};

struct InspectorFrontendClientSaveData {
    String displayType;
    String url;
    String content;
    bool base64Encoded;
};

class InspectorFrontendClient : public CanMakeWeakPtr<InspectorFrontendClient> {
public:
    enum class DockSide {
        Undocked = 0,
        Right,
        Left,
        Bottom,
    };

    virtual ~InspectorFrontendClient() = default;

    WEBCORE_EXPORT virtual void windowObjectCleared() = 0;
    virtual void frontendLoaded() = 0;

    virtual void pagePaused() = 0;
    virtual void pageUnpaused() = 0;

    virtual void startWindowDrag() = 0;
    virtual void moveWindowBy(float x, float y) = 0;

    // Information about the debuggable.
    virtual bool isRemote() const = 0;
    virtual String localizedStringsURL() const = 0;
    virtual String backendCommandsURL() const = 0;
    virtual Inspector::DebuggableType debuggableType() const = 0;
    virtual String targetPlatformName() const = 0;
    virtual String targetBuildVersion() const = 0;
    virtual String targetProductVersion() const = 0;
    virtual bool targetIsSimulator() const = 0;
    virtual unsigned inspectionLevel() const = 0;

    virtual void bringToFront() = 0;
    virtual void closeWindow() = 0;
    virtual void reopen() = 0;
    virtual void resetState() = 0;

    using Appearance = WebCore::InspectorFrontendClientAppearance;

    WEBCORE_EXPORT virtual void setForcedAppearance(Appearance) = 0;

    virtual UserInterfaceLayoutDirection userInterfaceLayoutDirection() const = 0;

    WEBCORE_EXPORT virtual bool supportsDockSide(DockSide) = 0;
    WEBCORE_EXPORT virtual void requestSetDockSide(DockSide) = 0;
    WEBCORE_EXPORT virtual void changeAttachedWindowHeight(unsigned) = 0;
    WEBCORE_EXPORT virtual void changeAttachedWindowWidth(unsigned) = 0;

    WEBCORE_EXPORT virtual void changeSheetRect(const FloatRect&) = 0;

    WEBCORE_EXPORT virtual void openURLExternally(const String& url) = 0;
    WEBCORE_EXPORT virtual void revealFileExternally(const String& path) = 0;

    // Keep in sync with `WI.FileUtilities.SaveMode` and `InspectorFrontendHost::SaveMode`.
    enum class SaveMode : uint8_t {
        SingleFile,
        FileVariants,
    };

    using SaveData = InspectorFrontendClientSaveData;

    virtual bool canSave(SaveMode) = 0;
    virtual void save(Vector<SaveData>&&, bool forceSaveAs) = 0;

    virtual bool canLoad() = 0;
    virtual void load(const String& path, CompletionHandler<void(const String&)>&&) = 0;

    virtual bool canPickColorFromScreen() = 0;
    virtual void pickColorFromScreen(CompletionHandler<void(const std::optional<WebCore::Color>&)>&&) = 0;

    virtual void inspectedURLChanged(const String&) = 0;
    virtual void showCertificate(const CertificateInfo&) = 0;

    virtual void setInspectorPageDeveloperExtrasEnabled(bool) = 0;

#if ENABLE(INSPECTOR_TELEMETRY)
    virtual bool supportsDiagnosticLogging() { return false; }
    virtual bool diagnosticLoggingAvailable() { return false; }
    virtual void logDiagnosticEvent(const String& /* eventName */, const DiagnosticLoggingClient::ValueDictionary&) { }
#endif

#if ENABLE(INSPECTOR_EXTENSIONS)
    virtual bool supportsWebExtensions() { return false; }
    virtual void didShowExtensionTab(const Inspector::ExtensionID&, const Inspector::ExtensionTabID&, const FrameIdentifier&) { }
    virtual void didHideExtensionTab(const Inspector::ExtensionID&, const Inspector::ExtensionTabID&) { }
    virtual void didNavigateExtensionTab(const Inspector::ExtensionID&, const Inspector::ExtensionTabID&, const URL&) { }
    virtual void inspectedPageDidNavigate(const URL&) { }
#endif

    WEBCORE_EXPORT virtual void sendMessageToBackend(const String&) = 0;
    WEBCORE_EXPORT virtual InspectorFrontendAPIDispatcher& frontendAPIDispatcher() = 0;
    WEBCORE_EXPORT virtual Page* frontendPage() = 0;

    WEBCORE_EXPORT virtual bool isUnderTest() = 0;
};

} // namespace WebCore
