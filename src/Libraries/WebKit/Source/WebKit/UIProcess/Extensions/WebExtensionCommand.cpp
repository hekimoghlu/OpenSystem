/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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
#import "WebExtensionCommand.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "WebExtensionCommandParameters.h"
#import "WebExtensionContext.h"
#import "WebExtensionContextProxyMessages.h"

namespace WebKit {

WebExtensionCommand::WebExtensionCommand(WebExtensionContext& extensionContext, const WebExtension::CommandData& data)
    : m_extensionContext(extensionContext)
    , m_identifier(data.identifier)
    , m_description(data.description)
    , m_activationKey(data.activationKey)
    , m_modifierFlags(data.modifierFlags)
{
}

bool WebExtensionCommand::operator==(const WebExtensionCommand& other) const
{
    return this == &other || (m_extensionContext == other.m_extensionContext && m_identifier == other.m_identifier);
}

bool WebExtensionCommand::isActionCommand() const
{
    RefPtr context = extensionContext();
    if (!context)
        return false;

    if (context->extension().supportsManifestVersion(3))
        return identifier() == "_execute_action"_s;

    return identifier() == "_execute_browser_action"_s || identifier() == "_execute_page_action"_s;
}

WebExtensionCommandParameters WebExtensionCommand::parameters() const
{
    return {
        identifier(),
        description(),
        shortcutString()
    };
}

WebExtensionContext* WebExtensionCommand::extensionContext() const
{
    return m_extensionContext.get();
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
