/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 1, 2025.
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

#include "Actions.h"
#include "Capabilities.h"
#include "SessionHost.h"
#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/HashSet.h>
#include <wtf/JSONValues.h>
#include <wtf/OptionSet.h>
#include <wtf/RefCounted.h>
#include <wtf/StdLibExtras.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

#if ENABLE(WEBDRIVER_BIDI)
#include "WebSocketServer.h"
#endif

namespace WebDriver {

class CommandResult;
class SessionHost;

class Session :
#if ENABLE(WEBDRIVER_BIDI)
public BiDiEventHandler // Inherits RefCounted
#else
public RefCounted<Session>
#endif
{
public:
    static Ref<Session> create(Ref<SessionHost>&& host)
    {
        return adoptRef(*new Session(WTFMove(host)));
    }
#if ENABLE(WEBDRIVER_BIDI)
    static Ref<Session> create(Ref<SessionHost>&& host, WeakPtr<WebSocketServer> bidiServer)
    {
        return adoptRef(*new Session(WTFMove(host), WTFMove(bidiServer)));
    }
#endif
    virtual ~Session();

    const String& id() const;
    const Capabilities& capabilities() const;
    bool isConnected() const;
#if ENABLE(WEBDRIVER_BIDI)
    bool hasBiDiEnabled() const { return m_hasBiDiEnabled; };
    void setHasBiDiEnabled(bool flag) { m_hasBiDiEnabled = flag; }
#endif
    double scriptTimeout() const { return m_scriptTimeout; }
    double pageLoadTimeout() const { return m_pageLoadTimeout; }
    double implicitWaitTimeout() const { return m_implicitWaitTimeout; }
    static const String& webElementIdentifier();
    static const String& shadowRootIdentifier();

    enum class FindElementsMode { Single,
        Multiple };
    enum class ExecuteScriptMode { Sync,
        Async };
    enum class ElementIsShadowRoot : bool { No,
        Yes };

    struct Cookie {
        String name;
        String value;
        std::optional<String> path;
        std::optional<String> domain;
        std::optional<bool> secure;
        std::optional<bool> httpOnly;
        std::optional<uint64_t> expiry;
        std::optional<String> sameSite;
    };

    InputSource& getOrCreateInputSource(const String& id, InputSource::Type, std::optional<PointerType>);

    void waitForNavigationToComplete(Function<void(CommandResult&&)>&&);
    void createTopLevelBrowsingContext(Function<void(CommandResult&&)>&&);
    void close(Function<void(CommandResult&&)>&&);
    void getTimeouts(Function<void(CommandResult&&)>&&);
    void setTimeouts(const Timeouts&, Function<void(CommandResult&&)>&&);

    void go(const String& url, Function<void(CommandResult&&)>&&);
    void getCurrentURL(Function<void(CommandResult&&)>&&);
    void back(Function<void(CommandResult&&)>&&);
    void forward(Function<void(CommandResult&&)>&&);
    void refresh(Function<void(CommandResult&&)>&&);
    void getTitle(Function<void(CommandResult&&)>&&);
    void getWindowHandle(Function<void(CommandResult&&)>&&);
    void closeWindow(Function<void(CommandResult&&)>&&);
    void switchToWindow(const String& windowHandle, Function<void(CommandResult&&)>&&);
    void getWindowHandles(Function<void(CommandResult&&)>&&);
    void newWindow(std::optional<String> typeHint, Function<void(CommandResult&&)>&&);
    void switchToFrame(RefPtr<JSON::Value>&&, Function<void(CommandResult&&)>&&);
    void switchToParentFrame(Function<void(CommandResult&&)>&&);
    void getWindowRect(Function<void(CommandResult&&)>&&);
    void setWindowRect(std::optional<double> x, std::optional<double> y, std::optional<double> width, std::optional<double> height, Function<void(CommandResult&&)>&&);
    void maximizeWindow(Function<void(CommandResult&&)>&&);
    void minimizeWindow(Function<void(CommandResult&&)>&&);
    void fullscreenWindow(Function<void(CommandResult&&)>&&);
    void findElements(const String& strategy, const String& selector, FindElementsMode, const String& rootElementID, ElementIsShadowRoot, Function<void(CommandResult&&)>&&);
    void getActiveElement(Function<void(CommandResult&&)>&&);
    void getElementShadowRoot(const String& elementID, Function<void(CommandResult&&)>&&);
    void isElementSelected(const String& elementID, Function<void(CommandResult&&)>&&);
    void getElementAttribute(const String& elementID, const String& attribute, Function<void(CommandResult&&)>&&);
    void getElementProperty(const String& elementID, const String& attribute, Function<void(CommandResult&&)>&&);
    void getElementCSSValue(const String& elementID, const String& cssProperty, Function<void(CommandResult&&)>&&);
    void getElementText(const String& elementID, Function<void(CommandResult&&)>&&);
    void getElementTagName(const String& elementID, Function<void(CommandResult&&)>&&);
    void getElementRect(const String& elementID, Function<void(CommandResult&&)>&&);
    void isElementEnabled(const String& elementID, Function<void(CommandResult&&)>&&);
    void getComputedRole(const String& elementID, Function<void(CommandResult&&)>&&);
    void getComputedLabel(const String& elementID, Function<void(CommandResult&&)>&&);
    void isElementDisplayed(const String& elementID, Function<void(CommandResult&&)>&&);
    void elementClick(const String& elementID, Function<void(CommandResult&&)>&&);
    void elementClear(const String& elementID, Function<void(CommandResult&&)>&&);
    void elementSendKeys(const String& elementID, const String& text, Function<void(CommandResult&&)>&&);
    void getPageSource(Function<void(CommandResult&&)>&&);
    void executeScript(const String& script, RefPtr<JSON::Array>&& arguments, ExecuteScriptMode, Function<void(CommandResult&&)>&&);
    void getAllCookies(Function<void(CommandResult&&)>&&);
    void getNamedCookie(const String& name, Function<void(CommandResult&&)>&&);
    void addCookie(const Cookie&, Function<void(CommandResult&&)>&&);
    void deleteCookie(const String& name, Function<void(CommandResult&&)>&&);
    void deleteAllCookies(Function<void(CommandResult&&)>&&);
    void performActions(Vector<Vector<Action>>&&, Function<void(CommandResult&&)>&&);
    void releaseActions(Function<void(CommandResult&&)>&&);
    void dismissAlert(Function<void(CommandResult&&)>&&);
    void acceptAlert(Function<void(CommandResult&&)>&&);
    void getAlertText(Function<void(CommandResult&&)>&&);
    void sendAlertText(const String&, Function<void(CommandResult&&)>&&);
    void takeScreenshot(std::optional<String> elementID, std::optional<bool> scrollIntoView, Function<void(CommandResult&&)>&&);

#if ENABLE(WEBDRIVER_BIDI)
    void enableGlobalEvent(const String&);
    void disableGlobalEvent(const String&);
    void dispatchEvent(RefPtr<JSON::Object>&&);
#endif

private:
    Session(Ref<SessionHost>&&);
#if ENABLE(WEBDRIVER_BIDI)
    Session(Ref<SessionHost>&&, WeakPtr<WebSocketServer>&&);
#endif

    void switchToTopLevelBrowsingContext(const String&);
    void switchToBrowsingContext(const String&, Function<void(CommandResult&&)>&&);
    void switchToBrowsingContext(const String& toplevelBrowsingContext, const String& browsingContext, Function<void(CommandResult&&)>&&);
    void closeTopLevelBrowsingContext(const String& toplevelBrowsingContext, Function<void(CommandResult&&)>&&);
    void closeAllToplevelBrowsingContexts(const String& toplevelBrowsingContext, Function<void(CommandResult&&)>&&);

    void getToplevelBrowsingContextRect(Function<void(CommandResult&&)>&&);

    std::optional<String> pageLoadStrategyString() const;

    void handleUserPrompts(Function<void(CommandResult&&)>&&);
    void handleUnexpectedAlertOpen(Function<void(CommandResult&&)>&&);
    void dismissAndNotifyAlert(Function<void(CommandResult&&)>&&);
    void acceptAndNotifyAlert(Function<void(CommandResult&&)>&&);
    void reportUnexpectedAlertOpen(Function<void(CommandResult&&)>&&);

    RefPtr<JSON::Object> createElement(RefPtr<JSON::Value>&&);
    Ref<JSON::Object> createElement(const String& elementID);
    RefPtr<JSON::Object> createShadowRoot(RefPtr<JSON::Value>&&);
    RefPtr<JSON::Object> extractElement(JSON::Value&);
    String extractElementID(JSON::Value&);
    Ref<JSON::Value> handleScriptResult(Ref<JSON::Value>&&);
    void elementIsEditable(const String& elementID, Function<void(CommandResult&&)>&&);

    struct Point {
        int x { 0 };
        int y { 0 };
    };

    struct Size {
        int width { 0 };
        int height { 0 };
    };

    struct Rect {
        Point origin;
        Size size;
    };

    enum class ElementLayoutOption {
        ScrollIntoViewIfNeeded = 1 << 0,
        UseViewportCoordinates = 1 << 1,
    };
    void computeElementLayout(const String& elementID, OptionSet<ElementLayoutOption>, Function<void(std::optional<Rect>&&, std::optional<Point>&&, bool, RefPtr<JSON::Object>&&)>&&);

    void elementIsFileUpload(const String& elementID, Function<void(CommandResult&&)>&&);
    enum class FileUploadType { Single,
        Multiple };
    std::optional<FileUploadType> parseElementIsFileUploadResult(const RefPtr<JSON::Value>&);
    void selectOptionElement(const String& elementID, Function<void(CommandResult&&)>&&);
    void setInputFileUploadFiles(const String& elementID, const String& text, bool multiple, Function<void(CommandResult&&)>&&);
    void didSetInputFileUploadFiles(bool wasCancelled);

    enum class MouseInteraction { Move,
        Down,
        Up,
        SingleClick,
        DoubleClick };
    void performMouseInteraction(int x, int y, MouseButton, MouseInteraction, Function<void(CommandResult&&)>&&);

    enum class KeyboardInteractionType { KeyPress,
        KeyRelease,
        InsertByKey };
    struct KeyboardInteraction {
        KeyboardInteractionType type { KeyboardInteractionType::InsertByKey };
        std::optional<String> text;
        std::optional<String> key;
    };
    enum KeyModifier {
        None = 0,
        Shift = 1 << 0,
        Control = 1 << 1,
        Alternate = 1 << 2,
        Meta = 1 << 3,
    };
    String virtualKeyForKey(UChar, KeyModifier&);
    void performKeyboardInteractions(Vector<KeyboardInteraction>&&, Function<void(CommandResult&&)>&&);

    struct InputSourceState {
        enum class Type { Null,
            Key,
            Pointer };

        Type type;
        String subtype;
        std::optional<MouseButton> pressedButton;
        std::optional<String> pressedKey;
        HashSet<String> pressedVirtualKeys;
    };
    InputSourceState& inputSourceState(const String& id);

    RefPtr<SessionHost> m_host;
    double m_scriptTimeout;
    double m_pageLoadTimeout;
    double m_implicitWaitTimeout;
    std::optional<String> m_toplevelBrowsingContext;
    std::optional<String> m_currentBrowsingContext;
    std::optional<String> m_currentParentBrowsingContext;
    HashMap<String, InputSource> m_activeInputSources;
    HashMap<String, InputSourceState> m_inputStateTable;
#if ENABLE(WEBDRIVER_BIDI)
    bool m_hasBiDiEnabled { false };

    // https://w3c.github.io/webdriver-bidi/#events
    HashSet<String> m_globalEventSet;
    WeakPtr<WebSocketServer> m_bidiServer;

    bool eventIsEnabled(const String&, const Vector<String>&);
    void emitEvent(const String&, RefPtr<JSON::Object>&&);
    String toInternalEventName(const String&);

    // Actual event handlers
    void doLogEntryAdded(RefPtr<JSON::Object>&&);
#endif
};

} // WebDriver
