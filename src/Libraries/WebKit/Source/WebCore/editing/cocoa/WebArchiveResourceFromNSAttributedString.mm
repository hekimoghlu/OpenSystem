/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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
#import "WebArchiveResourceFromNSAttributedString.h"

#import "ArchiveResource.h"
#import "MIMETypeRegistry.h"

using namespace WebCore;

@implementation WebArchiveResourceFromNSAttributedString

- (instancetype)initWithData:(NSData *)data URL:(NSURL *)URL MIMEType:(NSString *)MIMEType textEncodingName:(NSString *)textEncodingName frameName:(NSString *)frameName
{
    if (!(self = [super init]))
        return nil;

    if (!data || !URL || !MIMEType) {
        [self release];
        return nil;
    }

    if ([MIMEType isEqualToString:@"application/octet-stream"]) {
        // FIXME: This is a workaround for <rdar://problem/36074429>, and can be removed once that is fixed.
        auto mimeTypeFromURL = MIMETypeRegistry::mimeTypeForExtension(String(URL.pathExtension));
        if (!mimeTypeFromURL.isEmpty())
            MIMEType = mimeTypeFromURL;
    }

    resource = ArchiveResource::create(SharedBuffer::create(adoptNS([data copy]).get()), URL, MIMEType, textEncodingName, frameName, { });
    if (!resource) {
        [self release];
        return nil;
    }

    return self;
}

- (NSString *)MIMEType
{
    return resource->mimeType();
}

- (NSURL *)URL
{
    return resource->url();
}

@end
