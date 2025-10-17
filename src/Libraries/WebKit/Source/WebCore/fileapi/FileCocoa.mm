/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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
#import "File.h"

#if ENABLE(FILE_REPLACEMENT)

#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#import <wtf/FileSystem.h>
#include <wtf/text/MakeString.h>

#if PLATFORM(IOS_FAMILY)
#import <MobileCoreServices/MobileCoreServices.h>
#endif

namespace WebCore {

bool File::shouldReplaceFile(const String& path)
{
    if (path.isEmpty())
        return false;

    NSError *error;
    NSURL *pathURL = [NSURL URLByResolvingAliasFileAtURL:[NSURL fileURLWithPath:path isDirectory:NO] options:NSURLBookmarkResolutionWithoutUI error:&error];
    if (!pathURL) {
        LOG_ERROR("Failed to resolve alias at path %s with error %@.\n", path.utf8().data(), error);
        return false;
    }

    UTType *uti;
    if (![pathURL getResourceValue:&uti forKey:NSURLContentTypeKey error:&error]) {
        LOG_ERROR("Failed to get type identifier of resource at URL %@ with error %@.\n", pathURL, error);
        return false;
    }

    return [uti conformsToType:UTTypePackage];
}

void File::computeNameAndContentTypeForReplacedFile(const String& path, const String& nameOverride, String& effectiveName, String& effectiveContentType)
{
    ASSERT(!FileSystem::pathFileName(path).endsWith('/')); // Expecting to get a path without trailing slash, even for directories.
    effectiveContentType = "application/zip"_s;
    effectiveName = makeString((nameOverride.isNull() ? FileSystem::pathFileName(path) : nameOverride), ".zip"_s);
}

}

#endif // ENABLE(FILE_REPLACEMENT)
