/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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
#import <wtf/FileSystem.h>

#if PLATFORM(MAC)

#import <wtf/cocoa/NSURLExtras.h>
#import <wtf/spi/mac/MetadataSPI.h>
#import <wtf/text/WTFString.h>

namespace WTF {

void FileSystem::setMetadataURL(const String& path, const String& metadataURLString, const String& referrer)
{
    String urlString;
    if (NSURL *url = URLWithUserTypedString(metadataURLString))
        urlString = userVisibleString(URLByRemovingUserInfo(url));
    else
        urlString = metadataURLString;

    // Call Metadata API on a background queue because it can take some time.
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), [path = path.isolatedCopy(), urlString = WTFMove(urlString).isolatedCopy(), referrer = referrer.isolatedCopy()] {
        auto item = adoptCF(MDItemCreate(kCFAllocatorDefault, path.createCFString().get()));
        if (!item)
            return;

        auto whereFromAttribute = adoptNS([[NSMutableArray alloc] initWithObjects:urlString, nil]);
        if (!referrer.isNull())
            [whereFromAttribute addObject:referrer];

        MDItemSetAttribute(item.get(), kMDItemWhereFroms, (__bridge CFArrayRef)whereFromAttribute.get());
        MDItemSetAttribute(item.get(), kMDItemDownloadedDate, (__bridge CFArrayRef)@[ [NSDate date] ]);
    });
}

} // namespace WTF

#endif // PLATFORM(MAC)
