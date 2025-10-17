/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 2, 2024.
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
#import <WebKitLegacy/WebNSFileManagerExtras.h>

#import "WebKitNSStringExtras.h"
#import "WebNSURLExtras.h"
#import <WebCore/LoaderNSURLExtras.h>
#import <sys/stat.h>
#import <wtf/Assertions.h>
#import <wtf/FileSystem.h>

@implementation NSFileManager (WebNSFileManagerExtras)

#if !PLATFORM(IOS_FAMILY)

- (void)_webkit_setMetadataURL:(NSString *)URLString referrer:(NSString *)referrer atPath:(NSString *)path
{
    ASSERT(URLString);
    ASSERT(path);
    FileSystem::setMetadataURL(path, URLString, referrer);
}

#endif // !PLATFORM(IOS_FAMILY)

// -[NSFileManager fileExistsAtPath:] returns NO if there is a broken symlink at the path.
// So we use this function instead, which returns YES if there is anything there, including
// a broken symlink.
static BOOL fileExists(NSString *path)
{
    struct stat statBuffer;
    return !lstat([path fileSystemRepresentation], &statBuffer);
}

- (NSString *)_webkit_pathWithUniqueFilenameForPath:(NSString *)path
{
    // "Fix" the filename of the path.
    NSString *filename = filenameByFixingIllegalCharacters([path lastPathComponent]);
    path = [[path stringByDeletingLastPathComponent] stringByAppendingPathComponent:filename];

    if (fileExists(path)) {
        // Don't overwrite existing file by appending "-n", "-n.ext" or "-n.ext.ext" to the filename.
        NSString *extensions = nil;
        NSString *pathWithoutExtensions;
        NSString *lastPathComponent = [path lastPathComponent];
        NSRange periodRange = [lastPathComponent rangeOfString:@"."];
        
        if (periodRange.location == NSNotFound) {
            pathWithoutExtensions = path;
        } else {
            extensions = [lastPathComponent substringFromIndex:periodRange.location + 1];
            lastPathComponent = [lastPathComponent substringToIndex:periodRange.location];
            pathWithoutExtensions = [[path stringByDeletingLastPathComponent] stringByAppendingPathComponent:lastPathComponent];
        }

        for (unsigned i = 1; ; i++) {
            NSString *pathWithAppendedNumber = [NSString stringWithFormat:@"%@-%d", pathWithoutExtensions, i];
            path = [extensions length] ? [pathWithAppendedNumber stringByAppendingPathExtension:extensions] : pathWithAppendedNumber;
            if (!fileExists(path))
                break;
        }
    }

    return path;
}

#if PLATFORM(IOS_FAMILY)
- (NSString *)_webkit_createTemporaryDirectoryWithTemplatePrefix:(NSString *)prefix
{
    return FileSystem::createTemporaryDirectory(prefix);
}
#endif

@end

