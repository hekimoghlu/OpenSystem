/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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
#import "Icon.h"

#if PLATFORM(MAC)

#import "GraphicsContext.h"
#import "IntRect.h"
#import "UTIUtilities.h"
#import <AVFoundation/AVFoundation.h>
#import <wtf/RefPtr.h>
#import <wtf/text/WTFString.h>

#import <pal/cf/CoreMediaSoftLink.h>
#import <pal/cocoa/AVFoundationSoftLink.h>

namespace WebCore {

// FIXME: Move the code to ChromeClient::iconForFiles().
RefPtr<Icon> Icon::createIconForFiles(const Vector<String>& filenames)
{
    if (filenames.isEmpty())
        return nullptr;

    bool useIconFromFirstFile;
    useIconFromFirstFile = filenames.size() == 1;
    if (useIconFromFirstFile) {
        // Don't pass relative filenames -- we don't want a result that depends on the current directory.
        // Need 0U here to disambiguate String::operator[] from operator(NSString*, int)[]
        if (filenames[0].isEmpty() || filenames[0][0U] != '/')
            return nullptr;

        NSImage *image = [[NSWorkspace sharedWorkspace] iconForFile:filenames[0]];
        if (!image)
            return nullptr;

        return adoptRef(new Icon(image));
    }
    NSImage *image = [NSImage imageNamed:NSImageNameMultipleDocuments];
    if (!image)
        return nullptr;

    return adoptRef(new Icon(image));
}

RefPtr<Icon> Icon::createIconForFileExtension(const String& fileExtension)
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    NSImage *image = [[NSWorkspace sharedWorkspace] iconForFileType:[@"." stringByAppendingString:fileExtension]];
ALLOW_DEPRECATED_DECLARATIONS_END
    if (!image)
        return nullptr;

    return adoptRef(new Icon(image));
}

RefPtr<Icon> Icon::createIconForUTI(const String& UTI)
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    NSImage *image = [[NSWorkspace sharedWorkspace] iconForFileType:UTI];
ALLOW_DEPRECATED_DECLARATIONS_END
    if (!image)
        return nullptr;

    return adoptRef(new Icon(image));
}

}

#endif // PLATFORM(MAC)
