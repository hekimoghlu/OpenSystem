/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 9, 2023.
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

#include "InspectorFrontendAPIDispatcher.h"
#include "InspectorFrontendClient.h"
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Color;
class FloatRect;
class InspectorController;
class InspectorBackendDispatchTask;
class InspectorFrontendHost;
class LocalFrame;
class Page;

class InspectorFrontendClientLocal : public InspectorFrontendClient {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(InspectorFrontendClientLocal, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(InspectorFrontendClientLocal);
public:
    class WEBCORE_EXPORT Settings {
        WTF_MAKE_TZONE_ALLOCATED_EXPORT(Settings, WEBCORE_EXPORT);
    public:
        Settings() = default;
        virtual ~Settings() = default;
        virtual String getProperty(const String& name);
        virtual void setProperty(const String& name, const String& value);
        virtual void deleteProperty(const String& name);
    };

    WEBCORE_EXPORT InspectorFrontendClientLocal(InspectorController* inspectedPageController, Page* frontendPage, std::unique_ptr<Settings>);
    WEBCORE_EXPORT ~InspectorFrontendClientLocal() override;

    WEBCORE_EXPORT void resetState() override;

    WEBCORE_EXPORT void windowObjectCleared() final;
    WEBCORE_EXPORT void frontendLoaded() override;
    WEBCORE_EXPORT void pagePaused() final;
    WEBCORE_EXPORT void pageUnpaused() final;

    void startWindowDrag() override { }
    WEBCORE_EXPORT void moveWindowBy(float x, float y) final;

    WEBCORE_EXPORT UserInterfaceLayoutDirection userInterfaceLayoutDirection() const final;

    WEBCORE_EXPORT void requestSetDockSide(DockSide) final;
    WEBCORE_EXPORT void changeAttachedWindowHeight(unsigned) final;
    WEBCORE_EXPORT void changeAttachedWindowWidth(unsigned) final;
    WEBCORE_EXPORT void changeSheetRect(const FloatRect&) final;
    WEBCORE_EXPORT void openURLExternally(const String& url) final;
    void revealFileExternally(const String&) override { }
    bool canSave(InspectorFrontendClient::SaveMode) override { return false; }
    void save(Vector<InspectorFrontendClient::SaveData>&&, bool /* forceSaveAs */) override { }
    bool canLoad()  override { return false; }
    void load(const String&, CompletionHandler<void(const String&)>&& completionHandler) override { completionHandler(nullString()); }

    bool canPickColorFromScreen() override { return false; }
    void pickColorFromScreen(CompletionHandler<void(const std::optional<WebCore::Color>&)>&& completionHandler) override { completionHandler({ }); }

    void setInspectorPageDeveloperExtrasEnabled(bool) override { };

    virtual void attachWindow(DockSide) = 0;
    virtual void detachWindow() = 0;

    WEBCORE_EXPORT void sendMessageToBackend(const String& message) final;

    WEBCORE_EXPORT bool isUnderTest() final;
    bool isRemote() const final { return false; }
    WEBCORE_EXPORT unsigned inspectionLevel() const final;
    String backendCommandsURL() const final { return String(); };

    InspectorFrontendAPIDispatcher& frontendAPIDispatcher() final { return m_frontendAPIDispatcher; }
    WEBCORE_EXPORT Page* frontendPage() final;
    
    WEBCORE_EXPORT bool canAttachWindow();
    WEBCORE_EXPORT void setDockingUnavailable(bool);

    WEBCORE_EXPORT static unsigned constrainedAttachedWindowHeight(unsigned preferredHeight, unsigned totalWindowHeight);
    WEBCORE_EXPORT static unsigned constrainedAttachedWindowWidth(unsigned preferredWidth, unsigned totalWindowWidth);

    // Direct Frontend API
    WEBCORE_EXPORT bool isDebuggingEnabled();
    WEBCORE_EXPORT void setDebuggingEnabled(bool);

    WEBCORE_EXPORT bool isTimelineProfilingEnabled();
    WEBCORE_EXPORT void setTimelineProfilingEnabled(bool);

    WEBCORE_EXPORT bool isProfilingJavaScript();
    WEBCORE_EXPORT void startProfilingJavaScript();
    WEBCORE_EXPORT void stopProfilingJavaScript();

    WEBCORE_EXPORT void showConsole();

    WEBCORE_EXPORT void showMainResourceForFrame(LocalFrame*);

    WEBCORE_EXPORT void showResources();

    WEBCORE_EXPORT void setAttachedWindow(DockSide);

    WEBCORE_EXPORT Page* inspectedPage() const;

protected:
    virtual void setAttachedWindowHeight(unsigned) = 0;
    virtual void setAttachedWindowWidth(unsigned) = 0;
    WEBCORE_EXPORT void restoreAttachedWindowHeight();

    virtual void setSheetRect(const WebCore::FloatRect&) = 0;

private:
    friend class FrontendMenuProvider;
    std::optional<bool> evaluationResultToBoolean(InspectorFrontendAPIDispatcher::EvaluationResult);

    RefPtr<InspectorController> protectedInspectedPageController() const;

    WeakPtr<InspectorController> m_inspectedPageController;
    WeakPtr<Page> m_frontendPage;
    // TODO(yurys): this ref shouldn't be needed.
    RefPtr<InspectorFrontendHost> m_frontendHost;
    std::unique_ptr<InspectorFrontendClientLocal::Settings> m_settings;
    DockSide m_dockSide;
    Ref<InspectorBackendDispatchTask> m_dispatchTask;
    Ref<InspectorFrontendAPIDispatcher> m_frontendAPIDispatcher;
};

} // namespace WebCore
