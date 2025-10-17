/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 12, 2024.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "WebExtensionMenuItem.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "APIError.h"
#import "CocoaHelpers.h"
#import "WKNSError.h"
#import "WKWebExtensionContextPrivate.h"
#import "WebExtensionContext.h"
#import "WebExtensionContextProxyMessages.h"
#import "WebExtensionMenuItemContextParameters.h"
#import "WebExtensionMenuItemParameters.h"
#import <WebCore/LocalizedStrings.h>
#import <wtf/BlockPtr.h>
#import <wtf/cocoa/VectorCocoa.h>
#import <wtf/text/MakeString.h>

#if PLATFORM(IOS_FAMILY)
#import <UIKit/UIKit.h>
#endif

#if USE(APPKIT)
@implementation _WKWebExtensionMenuItem

- (instancetype)initWithTitle:(NSString *)title handler:(WebExtensionMenuItemHandlerBlock)handler
{
    RELEASE_ASSERT(handler);

    if (!(self = [super initWithTitle:title action:@selector(_performAction:) keyEquivalent:@""]))
        return nil;

    self.target = self;

    _handler = [handler copy];

    return self;
}

- (id)copyWithZone:(NSZone *)zone
{
    _WKWebExtensionMenuItem *copy = [super copyWithZone:zone];
    copy->_handler = [_handler copy];
    return copy;
}

- (IBAction)_performAction:(id)sender
{
    ASSERT(_handler);
    if (_handler)
        _handler(sender);
}

+ (BOOL)usesUserKeyEquivalents
{
    return NO;
}

@end
#endif // USE(APPKIT)

namespace WebKit {

WebExtensionMenuItem::WebExtensionMenuItem(WebExtensionContext& extensionContext, const WebExtensionMenuItemParameters& parameters)
    : m_extensionContext(extensionContext)
    , m_type(parameters.type.value_or(WebExtensionMenuItemType::Normal))
    , m_identifier(parameters.identifier)
    , m_title(removeAmpersands(parameters.title))
    , m_command(extensionContext.command(parameters.command))
    , m_checked(parameters.checked.value_or(false))
    , m_enabled(parameters.enabled.value_or(true))
    , m_visible(parameters.visible.value_or(true))
{
    relaxAdoptionRequirement();

    if (parameters.parentIdentifier) {
        if (RefPtr parentMenuItem = extensionContext.menuItem(parameters.parentIdentifier.value()))
            parentMenuItem->addSubmenuItem(*this);
    }

    if (parameters.contexts && !parameters.contexts.value().isEmpty())
        m_contexts = parameters.contexts.value();
    else if (m_parentMenuItem)
        m_contexts = m_parentMenuItem->contexts();
    else
        m_contexts = WebExtensionMenuItemContextType::Page;

    if (!parameters.iconsJSON.isEmpty()) {
        RefPtr parsedIcons = JSON::Value::parseJSON(parameters.iconsJSON);
        m_icons = parsedIcons ? parsedIcons->asObject() : nullptr;
#if ENABLE(WK_WEB_EXTENSIONS_ICON_VARIANTS)
        m_iconVariants = parsedIcons ? parsedIcons->asArray() : nullptr;
#endif
        clearIconCache();
    }

    if (parameters.documentURLPatterns) {
        for (auto& patternString : parameters.documentURLPatterns.value()) {
            if (RefPtr pattern = WebExtensionMatchPattern::getOrCreate(patternString))
                m_documentPatterns.add(pattern.releaseNonNull());
        }
    }

    if (parameters.targetURLPatterns) {
        for (auto& patternString : parameters.targetURLPatterns.value()) {
            if (RefPtr pattern = WebExtensionMatchPattern::getOrCreate(patternString))
                m_targetPatterns.add(pattern.releaseNonNull());
        }
    }
}

WebExtensionMenuItemParameters WebExtensionMenuItem::minimalParameters() const
{
    return {
        identifier(),
        parentMenuItem() ? std::optional { parentMenuItem()->identifier() } : std::nullopt,

        type(),

        nullString(), // title
        nullString(), // command
        nullString(), // iconsJSON

        isChecked(),
        isEnabled(),
        isVisible(),

        std::nullopt, // documentURLPatterns
        std::nullopt, // targetURLPatterns

        contexts(),
    };
}

void WebExtensionMenuItem::update(const WebExtensionMenuItemParameters& parameters)
{
    RefPtr extensionContext = this->extensionContext();
    ASSERT(extensionContext);

    if (parameters.type)
        m_type = parameters.type.value();

    if (!parameters.identifier.isEmpty())
        m_identifier = parameters.identifier;

    if (parameters.parentIdentifier) {
        RefPtr updatedParentMenuItem = extensionContext->menuItem(parameters.parentIdentifier.value());
        if (updatedParentMenuItem.get() != m_parentMenuItem) {
            if (RefPtr parentMenuItem = m_parentMenuItem.get())
                parentMenuItem->removeSubmenuItem(*this);

            if (updatedParentMenuItem)
                updatedParentMenuItem->addSubmenuItem(*this);
        }
    }

    if (!parameters.title.isEmpty())
        m_title = removeAmpersands(parameters.title);

    if (!parameters.command.isNull())
        m_command = extensionContext->command(parameters.command);

    if (!parameters.iconsJSON.isNull()) {
        RefPtr parsedIcons = JSON::Value::parseJSON(parameters.iconsJSON);
        m_icons = parsedIcons ? parsedIcons->asObject() : nullptr;
#if ENABLE(WK_WEB_EXTENSIONS_ICON_VARIANTS)
        m_iconVariants = parsedIcons ? parsedIcons->asArray() : nullptr;
#endif
        clearIconCache();
    }

    if (parameters.checked)
        m_checked = parameters.checked.value();

    if (parameters.enabled)
        m_enabled = parameters.enabled.value();

    if (parameters.visible)
        m_visible = parameters.visible.value();

    if (parameters.contexts) {
        if (!parameters.contexts.value().isEmpty())
            m_contexts = parameters.contexts.value();
        else if (m_parentMenuItem)
            m_contexts = m_parentMenuItem->contexts();
        else
            m_contexts = WebExtensionMenuItemContextType::Page;
    }

    if (parameters.documentURLPatterns) {
        m_documentPatterns.clear();

        for (auto& patternString : parameters.documentURLPatterns.value()) {
            if (RefPtr pattern = WebExtensionMatchPattern::getOrCreate(patternString))
                m_documentPatterns.add(pattern.releaseNonNull());
        }
    }

    if (parameters.targetURLPatterns) {
        m_targetPatterns.clear();

        for (auto& patternString : parameters.targetURLPatterns.value()) {
            if (RefPtr pattern = WebExtensionMatchPattern::getOrCreate(patternString))
                m_targetPatterns.add(pattern.releaseNonNull());
        }
    }
}

NSArray *WebExtensionMenuItem::matchingPlatformMenuItems(const MenuItemVector& menuItems, const WebExtensionMenuItemContextParameters& contextParameters, size_t limit)
{
    bool inRadioGroup = false;
    bool groupChecked = false;
    size_t count = 0;

    return createNSArray(menuItems, [&](auto& item) -> CocoaMenuItem * {
        if (limit && count >= limit)
            return nil;

        if (!item->matches(contextParameters))
            return nil;

        auto forceUnchecked = ForceUnchecked::No;

        switch (item->type()) {
        case WebExtensionMenuItemType::Radio:
            if (!inRadioGroup) {
                inRadioGroup = true;
                groupChecked = false;
            }

            if (item->isChecked()) {
                if (groupChecked)
                    forceUnchecked = ForceUnchecked::Yes;
                else
                    groupChecked = true;
            }

            break;

        default:
            inRadioGroup = false;
        }

        ++count;

        return item->platformMenuItem(contextParameters, forceUnchecked);
    }).get();
}

CocoaMenuItem *WebExtensionMenuItem::platformMenuItem(const WebExtensionMenuItemContextParameters& contextParameters, WebExtensionMenuItem::ForceUnchecked forceUnchecked) const
{
    ASSERT(extensionContext());
    ASSERT(matches(contextParameters));

    auto selectionString = !contextParameters.selectionString.isNull() ? WebCore::truncatedStringForMenuItem(contextParameters.selectionString) : emptyString();
    auto processedTitle = makeStringByReplacingAll(title(), "%s"_s, selectionString);
    auto *submenuItemArray = matchingPlatformMenuItems(submenuItems(), contextParameters);

#if USE(APPKIT)
    if (type() == WebExtensionMenuItemType::Separator)
        return [NSMenuItem separatorItem];

    auto *result = [[_WKWebExtensionMenuItem alloc] initWithTitle:processedTitle handler:makeBlockPtr([this, protectedThis = Ref { *this }, contextParameters](id sender) mutable {
        if (RefPtr context = extensionContext())
            context->performMenuItem(const_cast<WebExtensionMenuItem&>(*this), contextParameters, WebExtensionContext::UserTriggered::Yes);
    }).get()];

    if (RefPtr command = this->command()) {
        result.keyEquivalent = command->activationKey();
        result.keyEquivalentModifierMask = command->modifierFlags().toRaw();
    }

    auto idealSize = WebCore::FloatSize(16, 16);
    auto *image = toCocoaImage(icon(idealSize));

    image.size = idealSize;

    result.image = image;

    if (isCheckedType(type()))
        result.state = isChecked() ? NSControlStateValueOn : NSControlStateValueOff;

    result.enabled = isEnabled();
    result.hidden = !isVisible();

    if (submenuItems().isEmpty())
        return result;

    auto *submenu = [[NSMenu alloc] init];
    submenu.itemArray = submenuItemArray;
    result.submenu = submenu;

    return result;
#else
    // iOS does not support standalone separators.
    if (type() == WebExtensionMenuItemType::Separator)
        return nil;

    // iOS does not support sub-menus that are disabled or hidden, so return a normal action in that case.
    if (submenuItems().isEmpty() || !isEnabled() || !isVisible()) {
        auto *action = [UIAction actionWithTitle:processedTitle image:toCocoaImage(icon({ 20, 20 })) identifier:nil handler:makeBlockPtr([this, protectedThis = Ref { *this }, contextParameters](UIAction *) mutable {
            if (RefPtr context = extensionContext())
                context->performMenuItem(const_cast<WebExtensionMenuItem&>(*this), contextParameters, WebExtensionContext::UserTriggered::Yes);
        }).get()];

        if (isCheckedType(type()))
            action.state = isChecked() ? UIMenuElementStateOn : UIMenuElementStateOff;

        if (!isEnabled())
            action.attributes |= UIMenuElementAttributesDisabled;

        if (!isVisible())
            action.attributes |= UIMenuElementAttributesHidden;

        return action;
    }

    return [UIMenu menuWithTitle:processedTitle image:toCocoaImage(icon({ 20, 20 })) identifier:nil options:0 children:submenuItemArray];
#endif
}

RefPtr<WebCore::Icon> WebExtensionMenuItem::icon(WebCore::FloatSize idealSize) const
{
    RefPtr extensionContext = this->extensionContext();
    ASSERT(extensionContext);

#if ENABLE(WK_WEB_EXTENSIONS_ICON_VARIANTS)
    if (!m_iconVariants && !m_icons)
#else
    if (!m_icons)
#endif
        return nullptr;

    // Clear the cache if the display scales change (connecting display, etc.)
    auto currentScales = availableScreenScales();
    if (currentScales != m_cachedIconScales)
        clearIconCache();

    if (m_cachedIcon && CGSizeEqualToSize(idealSize, m_cachedIconIdealSize))
        return m_cachedIcon;

    RefPtr<WebCore::Icon> result;

#if ENABLE(WK_WEB_EXTENSIONS_ICON_VARIANTS)
    if (m_iconVariants) {
        result = extensionContext->protectedExtension()->bestIconVariant(m_iconVariants, WebCore::FloatSize(idealSize), [&](Ref<API::Error> error) {
            extensionContext->recordError(wrapper(error.get()));
        });
    } else
#endif // ENABLE(WK_WEB_EXTENSIONS_ICON_VARIANTS)
    if (m_icons) {
        result = extensionContext->protectedExtension()->bestIcon(m_icons, WebCore::FloatSize(idealSize), [&](Ref<API::Error> error) {
            extensionContext->recordError(wrapper(error.get()));
        });
    }

    if (result) {
        m_cachedIcon = result;
        m_cachedIconIdealSize = idealSize;
        m_cachedIconScales = currentScales;

        return result;
    }

    clearIconCache();

    return nullptr;
}

void WebExtensionMenuItem::clearIconCache() const
{
    m_cachedIcon = nil;
    m_cachedIconScales = { };
    m_cachedIconIdealSize = { };
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
