/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
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
#import "WebExtensionControllerConfiguration.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "SandboxUtilities.h"
#import "WKWebViewConfiguration.h"
#import <wtf/FileSystem.h>

namespace WebKit {

String WebExtensionControllerConfiguration::createStorageDirectoryPath(std::optional<WTF::UUID> identifier)
{
    String libraryPath = [NSFileManager.defaultManager URLForDirectory:NSLibraryDirectory inDomain:NSUserDomainMask appropriateForURL:nil create:NO error:nullptr].path;
    RELEASE_ASSERT(!libraryPath.isEmpty());

    String identifierPath = identifier ? identifier->toString().convertToASCIIUppercase() : "Default"_s;

    if (processHasContainer())
        return FileSystem::pathByAppendingComponents(libraryPath, { "WebKit"_s, "WebExtensions"_s, identifierPath });

    String appDirectoryName = NSBundle.mainBundle.bundleIdentifier ?: NSProcessInfo.processInfo.processName;
    return FileSystem::pathByAppendingComponents(libraryPath, { "WebKit"_s, appDirectoryName, "WebExtensions"_s, identifierPath });
}

String WebExtensionControllerConfiguration::createTemporaryStorageDirectoryPath()
{
    return FileSystem::createTemporaryDirectory(@"WebExtensions");
}

Ref<WebExtensionControllerConfiguration> WebExtensionControllerConfiguration::copy() const
{
    RefPtr<WebExtensionControllerConfiguration> result;

    if (m_identifier)
        result = create(m_identifier.value());
    else if (storageIsTemporary())
        result = createTemporary();
    else if (storageIsPersistent())
        result = createDefault();
    else
        result = createNonPersistent();

    result->setStorageDirectory(storageDirectory());
    result->setWebViewConfiguration([m_webViewConfiguration copy]);
    result->setDefaultWebsiteDataStore(m_defaultWebsiteDataStore.get());

    return result.releaseNonNull();
}

WKWebViewConfiguration *WebExtensionControllerConfiguration::webViewConfiguration()
{
    if (!m_webViewConfiguration)
        m_webViewConfiguration = [[WKWebViewConfiguration alloc] init];
    return m_webViewConfiguration.get();
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
