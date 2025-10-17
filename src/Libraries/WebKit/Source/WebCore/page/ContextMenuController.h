/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 11, 2022.
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

#if ENABLE(CONTEXT_MENUS)

#include "ContextMenuContext.h"
#include "ContextMenuItem.h"
#include "HitTestRequest.h"
#include <wtf/OptionSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakRef.h>

namespace WebCore {

class ContextMenuClient;
class ContextMenuProvider;
class Event;
class HitTestResult;
class Page;

class ContextMenuController {
    WTF_MAKE_TZONE_ALLOCATED(ContextMenuController);
public:
    ContextMenuController(Page&, UniqueRef<ContextMenuClient>&&);
    ~ContextMenuController();

    Page& page();
    ContextMenuClient& client() { return m_client.get(); }

    ContextMenu* contextMenu() const { return m_contextMenu.get(); }
    WEBCORE_EXPORT void clearContextMenu();

    void handleContextMenuEvent(Event&);
    void showContextMenu(Event&, ContextMenuProvider&);

    void populate();
    WEBCORE_EXPORT void didDismissContextMenu();
    WEBCORE_EXPORT void contextMenuItemSelected(ContextMenuAction, const String& title);
    void addDebuggingItems();

    WEBCORE_EXPORT void checkOrEnableIfNeeded(ContextMenuItem&) const;

    void setContextMenuContext(const ContextMenuContext& context) { m_context = context; }
    const ContextMenuContext& context() const { return m_context; }
    const HitTestResult& hitTestResult() const { return m_context.hitTestResult(); }

#if USE(ACCESSIBILITY_CONTEXT_MENUS)
    void showContextMenuAt(LocalFrame&, const IntPoint& clickPoint);
#endif
    
#if ENABLE(SERVICE_CONTROLS)
    void showImageControlsMenu(Event&);
#endif

private:
    std::unique_ptr<ContextMenu> maybeCreateContextMenu(Event&, OptionSet<HitTestRequest::Type> hitType, ContextMenuContext::Type);
    void showContextMenu(Event&);

    void appendItem(ContextMenuItem&, ContextMenu* parentMenu);

    void createAndAppendFontSubMenu(ContextMenuItem&);
    void createAndAppendSpellingAndGrammarSubMenu(ContextMenuItem&);
    void createAndAppendSpellingSubMenu(ContextMenuItem&);
    void createAndAppendSpeechSubMenu(ContextMenuItem&);
    void createAndAppendWritingDirectionSubMenu(ContextMenuItem&);
    void createAndAppendTextDirectionSubMenu(ContextMenuItem&);
    void createAndAppendSubstitutionsSubMenu(ContextMenuItem&);
    void createAndAppendTransformationsSubMenu(ContextMenuItem&);
    bool shouldEnableCopyLinkWithHighlight() const;
#if PLATFORM(GTK)
    void createAndAppendUnicodeSubMenu(ContextMenuItem&);
#endif

#if ENABLE(PDFJS)
    void performPDFJSAction(LocalFrame&, const String& action);
#endif

    WeakRef<Page> m_page;
    UniqueRef<ContextMenuClient> m_client;
    std::unique_ptr<ContextMenu> m_contextMenu;
    RefPtr<ContextMenuProvider> m_menuProvider;
    ContextMenuContext m_context;
    bool m_isHandlingContextMenuEvent { false };
};

} // namespace WebCore

#endif // ENABLE(CONTEXT_MENUS)
