/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 9, 2022.
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
#import "WebExtensionContext.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "CocoaHelpers.h"
#import "WebExtensionCommand.h"
#import "WebExtensionCommandParameters.h"
#import "WebExtensionContextProxyMessages.h"
#import "WebExtensionTab.h"
#import "WebExtensionUtilities.h"

namespace WebKit {

bool WebExtensionContext::isCommandsMessageAllowed()
{
    return isLoaded() && protectedExtension()->hasCommands();
}

void WebExtensionContext::commandsGetAll(CompletionHandler<void(Vector<WebExtensionCommandParameters>)>&& completionHandler)
{
    auto results = WTF::map(commands(), [](auto& command) {
        return command->parameters();
    });

    completionHandler(WTFMove(results));
}

void WebExtensionContext::fireCommandEventIfNeeded(const WebExtensionCommand& command, WebExtensionTab* tab)
{
    constexpr auto type = WebExtensionEventListenerType::CommandsOnCommand;
    wakeUpBackgroundContentIfNecessaryToFireEvents({ type }, [=, this, protectedThis = Ref { *this }, command = Ref { command }, tab = RefPtr { tab }] {
        sendToProcessesForEvent(type, Messages::WebExtensionContextProxy::DispatchCommandsCommandEvent(command->identifier(), tab ? std::optional(tab->parameters()) : std::nullopt));
    });
}

void WebExtensionContext::fireCommandChangedEventIfNeeded(const WebExtensionCommand& command, const String& oldShortcut)
{
    constexpr auto type = WebExtensionEventListenerType::CommandsOnChanged;
    wakeUpBackgroundContentIfNecessaryToFireEvents({ type }, [=, this, protectedThis = Ref { *this }, command = Ref { command }] {
        sendToProcessesForEvent(type, Messages::WebExtensionContextProxy::DispatchCommandsChangedEvent(command->identifier(), oldShortcut, command->shortcutString()));
    });
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
