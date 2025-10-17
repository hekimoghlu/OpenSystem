/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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
#import "LoaderNSURLExtras.h"

#import "LocalizedStrings.h"
#import "MIMETypeRegistry.h"
#import <wtf/Assertions.h>
#import <wtf/URL.h>
#import <wtf/Vector.h>
#import <wtf/text/WTFString.h>

using namespace WebCore;

NSString *suggestedFilenameWithMIMEType(NSURL *url, const String& mimeType)
{
    return suggestedFilenameWithMIMEType(url, mimeType, copyImageUnknownFileLabel());
}

NSString *suggestedFilenameWithMIMEType(NSURL *url, const String& mimeType, const String& defaultValue)
{
    // Get the filename from the URL. Try the lastPathComponent first.
    NSString *lastPathComponent = [[url path] lastPathComponent];
    NSString *filename = filenameByFixingIllegalCharacters(lastPathComponent);
    NSString *extension = nil;

    if ([filename length] == 0 || [lastPathComponent isEqualToString:@"/"]) {
        // lastPathComponent is no good, try the host.
        auto host = URL(url).host().createNSString();
        filename = filenameByFixingIllegalCharacters(host.get());
        if ([filename length] == 0) {
            // Can't make a filename using this URL, use the default value.
            filename = defaultValue;
        }
    } else {
        // Save the extension for later correction. Only correct the extension of the lastPathComponent.
        // For example, if the filename ends up being the host, we wouldn't want to correct ".com" in "www.apple.com".
        extension = [filename pathExtension];
    }

    if (!mimeType)
        return filename;

    // Do not correct filenames that are reported with a mime type of tar, and 
    // have a filename which has .tar in it or ends in .tgz
    if ((mimeType == "application/tar"_s || mimeType == "application/x-tar"_s)
        && (String(filename).containsIgnoringASCIICase(".tar"_s)
        || String(filename).endsWithIgnoringASCIICase(".tgz"_s))) {
        return filename;
    }

    // I don't think we need to worry about this for the image case
    // If the type is known, check the extension and correct it if necessary.
    if (mimeType != "application/octet-stream"_s && mimeType != "text/plain"_s) {
        Vector<String> extensions = MIMETypeRegistry::extensionsForMIMEType(mimeType);

        if (extensions.isEmpty() || !extensions.contains(String(extension))) {
            // The extension doesn't match the MIME type. Correct this.
            NSString *correctExtension = MIMETypeRegistry::preferredExtensionForMIMEType(mimeType);
            if ([correctExtension length] != 0) {
                // Append the correct extension.
                filename = [filename stringByAppendingPathExtension:correctExtension];
            }
        }
    }

    return filename;
}

NSString *filenameByFixingIllegalCharacters(NSString *string)
{
    RetainPtr filename = adoptNS([string mutableCopy]);

    // Strip null characters.
    unichar nullChar = 0;
    [filename replaceOccurrencesOfString:[NSString stringWithCharacters:&nullChar length:0] withString:@"" options:0 range:NSMakeRange(0, [filename length])];

    // Replace "/" with "-".
    [filename replaceOccurrencesOfString:@"/" withString:@"-" options:0 range:NSMakeRange(0, [filename length])];

    // Replace ":" with "-".
    [filename replaceOccurrencesOfString:@":" withString:@"-" options:0 range:NSMakeRange(0, [filename length])];

    // Strip leading dots.
    while ([filename hasPrefix:@"."])
        [filename deleteCharactersInRange:NSMakeRange(0, 1)];

    return filename.autorelease();
}
