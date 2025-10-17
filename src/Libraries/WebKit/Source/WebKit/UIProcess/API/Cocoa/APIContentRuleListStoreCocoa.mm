/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 6, 2022.
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
#import "APIContentRuleListStore.h"

#if ENABLE(CONTENT_EXTENSIONS)

#import "SandboxUtilities.h"
#import <WebCore/SharedBuffer.h>

namespace API {

WTF::String ContentRuleListStore::defaultStorePath()
{
    static dispatch_once_t onceToken;
    static NeverDestroyed<RetainPtr<NSURL>> contentRuleListStoreURL;

    dispatch_once(&onceToken, ^{
        NSURL *url = [[NSFileManager defaultManager] URLForDirectory:NSLibraryDirectory inDomain:NSUserDomainMask appropriateForURL:nullptr create:NO error:nullptr];
        if (!url)
            RELEASE_ASSERT_NOT_REACHED();

        url = [url URLByAppendingPathComponent:@"WebKit" isDirectory:YES];

        if (!WebKit::processHasContainer()) {
            NSString *bundleIdentifier = [NSBundle mainBundle].bundleIdentifier;
            if (!bundleIdentifier)
                bundleIdentifier = [NSProcessInfo processInfo].processName;
            url = [url URLByAppendingPathComponent:bundleIdentifier isDirectory:YES];
        }
        
        contentRuleListStoreURL.get() = [url URLByAppendingPathComponent:@"ContentRuleLists" isDirectory:YES];
    });

    if (![[NSFileManager defaultManager] createDirectoryAtURL:contentRuleListStoreURL.get().get() withIntermediateDirectories:YES attributes:nil error:nullptr])
        LOG_ERROR("Failed to create directory %@", contentRuleListStoreURL.get().get());

    return [contentRuleListStoreURL.get() absoluteURL].path;
}

} // namespace API

#endif // ENABLE(CONTENT_EXTENSIONS)
