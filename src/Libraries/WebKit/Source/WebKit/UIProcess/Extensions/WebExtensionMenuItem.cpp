/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 31, 2025.
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
#import "config.h"
#import "WebExtensionMenuItem.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "WebExtensionContext.h"
#import "WebExtensionContextProxyMessages.h"
#import "WebExtensionMatchPattern.h"
#import "WebExtensionMenuItemContextParameters.h"
#import "WebExtensionMenuItemParameters.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebExtensionMenuItem);

bool WebExtensionMenuItem::operator==(const WebExtensionMenuItem& other) const
{
    return this == &other || (m_extensionContext == other.m_extensionContext && m_identifier == other.m_identifier);
}

WebExtensionContext* WebExtensionMenuItem::extensionContext() const
{
    return m_extensionContext.get();
}

String WebExtensionMenuItem::removeAmpersands(const String& title)
{
    // The title may contain ampersands used to indicate shortcut keys for the item. We don't support keyboard
    // shortcuts this way, but we still remove the ampersands from the visible title of the item.

    StringBuilder stringBuilder;
    size_t startIndex = 0;

    while (startIndex < title.length()) {
        size_t ampersandPosition = title.find('&', startIndex);

        // If no ampersand is found, append the rest of the string and break.
        if (ampersandPosition == notFound) {
            stringBuilder.append(title.substring(startIndex));
            break;
        }

        // Append the part of the string before the ampersand.
        if (ampersandPosition > startIndex)
            stringBuilder.append(title.substring(startIndex, ampersandPosition - startIndex));

        // If a double ampersand is found, replace it with a single ampersand.
        if (ampersandPosition < title.length() - 1 && title[ampersandPosition + 1] == '&') {
            stringBuilder.append('&');
            startIndex = ampersandPosition + 2;
            continue;
        }

        // Skip the single ampersand.
        startIndex = ampersandPosition + 1;
    }

    return stringBuilder.toString();
}

bool WebExtensionMenuItem::matches(const WebExtensionMenuItemContextParameters& contextParameters) const
{
    if (!contexts().containsAny(contextParameters.types))
        return false;

    using ContextType = WebExtensionMenuItemContextType;

    auto matchesType = [&](const OptionSet<ContextType>& types) {
        for (auto type : types) {
            if (contextParameters.types.contains(type) && contexts().contains(type))
                return true;
        }

        return false;
    };

    auto matchesPattern = [&](const auto& patterns, const URL& url) {
        if (url.isNull() || patterns.isEmpty())
            return true;

        for (const auto& pattern : patterns) {
            if (pattern->matchesURL(url))
                return true;
        }

        return false;
    };

    // Document patterns match for any context type.
    if (!matchesPattern(documentPatterns(), contextParameters.frameURL))
        return false;

    if (matchesType({ ContextType::Action, ContextType::Tab })) {
        // Additional context checks are not required for Action or Tab.
        return true;
    }

    if (matchesType(ContextType::Link)) {
        ASSERT(!contextParameters.linkURL.isNull());
        if (!matchesPattern(targetPatterns(), contextParameters.linkURL))
            return false;
    }

    if (matchesType({ ContextType::Image, ContextType::Video, ContextType::Audio })) {
        ASSERT(!contextParameters.sourceURL.isNull());
        if (!matchesPattern(targetPatterns(), contextParameters.sourceURL))
            return false;
    }

    if (matchesType(ContextType::Selection) && contextParameters.selectionString.isEmpty())
        return false;

    if (matchesType(ContextType::Editable) && !contextParameters.editable)
        return false;

    return true;
}

bool WebExtensionMenuItem::toggleCheckedIfNeeded(const WebExtensionMenuItemContextParameters& contextParameters)
{
    ASSERT(extensionContext());

    bool wasChecked = isChecked();

    switch (type()) {
    case WebExtensionMenuItemType::Normal:
    case WebExtensionMenuItemType::Separator:
        ASSERT(!wasChecked);
        break;

    case WebExtensionMenuItemType::Checkbox:
        setChecked(!wasChecked);
        break;

    case WebExtensionMenuItemType::Radio:
        if (wasChecked)
            break;

        setChecked(true);

        auto& items = parentMenuItem() ? parentMenuItem()->submenuItems() : extensionContext()->mainMenuItems();

        auto index = items.find(*this);
        if (index == notFound)
            break;

        // Uncheck all radio items in the same group before the current item.
        for (ssize_t i = index - 1; i >= 0; --i) {
            auto& item = items[i];
            if (!item->matches(contextParameters))
                continue;

            if (item->type() != WebExtensionMenuItemType::Radio)
                break;

            item->setChecked(false);
        }

        // Uncheck all radio items in the same group after the current item.
        for (size_t i = index + 1; i < items.size(); ++i) {
            auto& item = items[i];
            if (!item->matches(contextParameters))
                continue;

            if (item->type() != WebExtensionMenuItemType::Radio)
                break;

            item->setChecked(false);
        }

        break;
    }

    return wasChecked;
}

void WebExtensionMenuItem::addSubmenuItem(WebExtensionMenuItem& menuItem)
{
    menuItem.m_parentMenuItem = this;
    m_submenuItems.append(menuItem);
}

void WebExtensionMenuItem::removeSubmenuItem(WebExtensionMenuItem& menuItem)
{
    menuItem.m_parentMenuItem = nullptr;
    m_submenuItems.removeAll(menuItem);
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
