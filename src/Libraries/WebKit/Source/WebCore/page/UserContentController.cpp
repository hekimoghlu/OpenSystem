/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 10, 2024.
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
#include "UserContentController.h"

#include "DOMWrapperWorld.h"
#include "UserScript.h"
#include "UserStyleSheet.h"
#include <JavaScriptCore/JSObjectInlines.h>

#if ENABLE(CONTENT_EXTENSIONS)
#include "CompiledContentExtension.h"
#endif

namespace WebCore {

Ref<UserContentController> UserContentController::create()
{
    return adoptRef(*new UserContentController);
}

UserContentController::UserContentController() = default;

UserContentController::~UserContentController() = default;

void UserContentController::forEachUserScript(Function<void(DOMWrapperWorld&, const UserScript&)>&& functor) const
{
    for (const auto& worldAndUserScriptVector : m_userScripts) {
        auto& world = *worldAndUserScriptVector.key.get();
        for (const auto& userScript : *worldAndUserScriptVector.value)
            functor(world, *userScript);
    }
}

void UserContentController::forEachUserStyleSheet(Function<void(const UserStyleSheet&)>&& functor) const
{
    for (auto& styleSheetVector : m_userStyleSheets.values()) {
        for (const auto& styleSheet : *styleSheetVector)
            functor(*styleSheet);
    }
}

#if ENABLE(USER_MESSAGE_HANDLERS)
void UserContentController::forEachUserMessageHandler(Function<void(const UserMessageHandlerDescriptor&)>&&) const
{
}
#endif

void UserContentController::addUserScript(DOMWrapperWorld& world, std::unique_ptr<UserScript> userScript)
{
    auto& scriptsInWorld = m_userScripts.ensure(&world, [&] { return makeUnique<UserScriptVector>(); }).iterator->value;
    scriptsInWorld->append(WTFMove(userScript));
}

void UserContentController::removeUserScript(DOMWrapperWorld& world, const URL& url)
{
    auto it = m_userScripts.find(&world);
    if (it == m_userScripts.end())
        return;

    auto scripts = it->value.get();
    for (int i = scripts->size() - 1; i >= 0; --i) {
        if (scripts->at(i)->url() == url)
            scripts->remove(i);
    }

    if (scripts->isEmpty())
        m_userScripts.remove(it);
}

void UserContentController::removeUserScripts(DOMWrapperWorld& world)
{
    m_userScripts.remove(&world);
}

void UserContentController::addUserStyleSheet(DOMWrapperWorld& world, std::unique_ptr<UserStyleSheet> userStyleSheet, UserStyleInjectionTime injectionTime)
{
    auto& styleSheetsInWorld = m_userStyleSheets.ensure(&world, [&] { return makeUnique<UserStyleSheetVector>(); }).iterator->value;
    styleSheetsInWorld->append(WTFMove(userStyleSheet));

    if (injectionTime == InjectInExistingDocuments)
        invalidateInjectedStyleSheetCacheInAllFramesInAllPages();
}

void UserContentController::removeUserStyleSheet(DOMWrapperWorld& world, const URL& url)
{
    auto it = m_userStyleSheets.find(&world);
    if (it == m_userStyleSheets.end())
        return;

    auto& stylesheets = *it->value;

    bool sheetsChanged = false;
    for (int i = stylesheets.size() - 1; i >= 0; --i) {
        if (stylesheets[i]->url() == url) {
            stylesheets.remove(i);
            sheetsChanged = true;
        }
    }

    if (!sheetsChanged)
        return;

    if (stylesheets.isEmpty())
        m_userStyleSheets.remove(it);

    invalidateInjectedStyleSheetCacheInAllFramesInAllPages();
}

void UserContentController::removeUserStyleSheets(DOMWrapperWorld& world)
{
    if (!m_userStyleSheets.remove(&world))
        return;

    invalidateInjectedStyleSheetCacheInAllFramesInAllPages();
}

void UserContentController::removeAllUserContent()
{
    m_userScripts.clear();

    if (!m_userStyleSheets.isEmpty()) {
        m_userStyleSheets.clear();
        invalidateInjectedStyleSheetCacheInAllFramesInAllPages();
    }
}

} // namespace WebCore
