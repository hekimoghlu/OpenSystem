/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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
#import "_WKWebsiteDataSizeInternal.h"

#import "WKWebsiteDataRecordInternal.h"

@implementation _WKWebsiteDataSize

- (instancetype)initWithSize:(const WebKit::WebsiteDataRecord::Size&)size
{
    if (!(self = [super init]))
        return nil;

    _size = size;

    return self;
}

- (unsigned long long)totalSize
{
    return _size.totalSize;
}

- (unsigned long long)sizeOfDataTypes:(NSSet *)dataTypes
{
    unsigned long long size = 0;

    for (NSString *dataType in dataTypes) {
        if (auto websiteDataType = WebKit::toWebsiteDataType(dataType))
            size += _size.typeSizes.get(static_cast<unsigned>(*websiteDataType));
    }
    
    return size;
}

@end
