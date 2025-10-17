/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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
#import <wtf/URL.h>

#import <wtf/URLParser.h>
#import <wtf/cf/CFURLExtras.h>
#import <wtf/cocoa/NSURLExtras.h>
#import <wtf/cocoa/TypeCastsCocoa.h>
#import <wtf/text/CString.h>

@interface NSString (WTFNSURLExtras)
- (BOOL)_web_looksLikeIPAddress;
@end

namespace WTF {

URL::URL(NSURL *cocoaURL)
    : URL(bridge_cast(cocoaURL))
{
}

URL::operator NSURL *() const
{
    // Creating a toll-free bridged CFURL because creation with NSURL methods would not preserve the original string.
    // We'll need fidelity when round-tripping via CFURLGetBytes().
    return createCFURL().bridgingAutorelease();
}

RetainPtr<CFURLRef> URL::emptyCFURL()
{
    // We use the toll-free bridge to create an empty value that is distinct from null that no CFURL function can create.
    // FIXME: When we originally wrote this, we thought that creating empty CF URLs was valuable; can we do without it now?
    return bridge_cast(adoptNS([[NSURL alloc] initWithString:@""]));
}

bool URL::hostIsIPAddress(StringView host)
{
    return [host.createNSStringWithoutCopying().get() _web_looksLikeIPAddress];
}

RetainPtr<id> makeNSArrayElement(const URL& vectorElement)
{
    return bridge_cast(vectorElement.createCFURL());
}

std::optional<URL> makeVectorElement(const URL*, id arrayElement)
{
    if (![arrayElement isKindOfClass:NSURL.class])
        return std::nullopt;
    return { { arrayElement } };
}

}
