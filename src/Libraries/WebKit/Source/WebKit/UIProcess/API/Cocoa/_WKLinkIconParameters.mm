/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 28, 2025.
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
#import "_WKLinkIconParametersInternal.h"

#import <WebCore/LinkIcon.h>

@implementation _WKLinkIconParameters {
    RetainPtr<NSURL> _url;
    WKLinkIconType _iconType;
    RetainPtr<NSString> _mimeType;
    RetainPtr<NSNumber> _size;
    RetainPtr<NSMutableDictionary> _attributes;
}

- (instancetype)_initWithLinkIcon:(const WebCore::LinkIcon&)linkIcon
{
    if (!(self = [super init]))
        return nil;

    _url = (NSURL *)linkIcon.url;
    _mimeType = (NSString *)linkIcon.mimeType;

    if (linkIcon.size)
        _size = adoptNS([[NSNumber alloc] initWithUnsignedInt:linkIcon.size.value()]);

    switch (linkIcon.type) {
    case WebCore::LinkIconType::Favicon:
        _iconType = WKLinkIconTypeFavicon;
        break;
    case WebCore::LinkIconType::TouchIcon:
        _iconType = WKLinkIconTypeTouchIcon;
        break;
    case WebCore::LinkIconType::TouchPrecomposedIcon:
        _iconType = WKLinkIconTypeTouchPrecomposedIcon;
        break;
    }

    _attributes = adoptNS([[NSMutableDictionary alloc] initWithCapacity:linkIcon.attributes.size()]);
    for (auto& attributePair : linkIcon.attributes)
        _attributes.get()[(NSString *)attributePair.first] = attributePair.second;

    return self;
}

- (NSURL *)url
{
    return _url.get();
}

- (NSString *)mimeType
{
    return _mimeType.get();
}

- (NSNumber *)size
{
    return _size.get();
}

- (WKLinkIconType)iconType
{
    return _iconType;
}

- (NSDictionary *)attributes
{
    return _attributes.get();
}

@end
