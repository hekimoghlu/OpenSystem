/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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
#import "WebExtensionAPICommands.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "CocoaHelpers.h"
#import "Logging.h"
#import "MessageSenderInlines.h"
#import "WebExtensionAPINamespace.h"
#import "WebExtensionAPITabs.h"
#import "WebExtensionCommandParameters.h"
#import "WebExtensionContextMessages.h"
#import "WebExtensionContextProxy.h"
#import "WebExtensionUtilities.h"
#import "WebFrame.h"
#import "WebProcess.h"
#import <WebCore/LocalFrame.h>

namespace WebKit {

static NSString * const nameKey = @"name";
static NSString * const descriptionKey = @"description";
static NSString * const shortcutKey = @"shortcut";
static NSString * const newShortcutKey = @"newShortcut";
static NSString * const oldShortcutKey = @"oldShortcut";

static inline NSDictionary *toAPI(const WebExtensionCommandParameters& command)
{
    return @{
        nameKey: (NSString *)command.identifier,
        descriptionKey: (NSString *)command.description,
        shortcutKey: (NSString *)command.shortcut
    };
}

static inline NSArray *toAPI(const Vector<WebExtensionCommandParameters>& commands)
{
    NSMutableArray *result = [NSMutableArray arrayWithCapacity:commands.size()];

    for (auto& command : commands)
        [result addObject:toAPI(command)];

    return [result copy];
}

void WebExtensionAPICommands::getAll(Ref<WebExtensionCallbackHandler>&& callback)
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/commands/getAll

    WebProcess::singleton().sendWithAsyncReply(Messages::WebExtensionContext::CommandsGetAll(), [protectedThis = Ref { *this }, callback = WTFMove(callback)](Vector<WebExtensionCommandParameters> commands) {
        callback->call(toAPI(commands));
    }, extensionContext().identifier());
}

WebExtensionAPIEvent& WebExtensionAPICommands::onCommand()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/commands/onCommand

    if (!m_onCommand)
        m_onCommand = WebExtensionAPIEvent::create(*this, WebExtensionEventListenerType::CommandsOnCommand);

    return *m_onCommand;
}

WebExtensionAPIEvent& WebExtensionAPICommands::onChanged()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/commands/onChanged

    if (!m_onChanged)
        m_onChanged = WebExtensionAPIEvent::create(*this, WebExtensionEventListenerType::CommandsOnChanged);

    return *m_onChanged;
}

void WebExtensionContextProxy::dispatchCommandsCommandEvent(const String& identifier, const std::optional<WebExtensionTabParameters>& tabParameters)
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/commands/onCommand

    auto *tab = tabParameters ? toWebAPI(tabParameters.value()) : nil;

    enumerateFramesAndNamespaceObjects([&](auto& frame, auto& namespaceObject) {
        RefPtr coreFrame = frame.protectedCoreLocalFrame();
        WebCore::UserGestureIndicator gestureIndicator(WebCore::IsProcessingUserGesture::Yes, coreFrame ? coreFrame->document() : nullptr);
        namespaceObject.commands().onCommand().invokeListenersWithArgument((NSString *)identifier, tab);
    });
}

void WebExtensionContextProxy::dispatchCommandsChangedEvent(const String& identifier, const String& oldShortcut, const String& newShortcut)
{
    auto *changeInfo = @{
        nameKey: (NSString *)identifier,
        oldShortcutKey: (NSString *)oldShortcut,
        newShortcutKey: (NSString *)newShortcut
    };

    enumerateNamespaceObjects([&](auto& namespaceObject) {
        // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/commands/onChanged
        namespaceObject.commands().onChanged().invokeListenersWithArgument(changeInfo);
    });
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
