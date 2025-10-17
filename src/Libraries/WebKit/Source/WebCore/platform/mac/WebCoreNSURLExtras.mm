/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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
#import "WebCoreNSURLExtras.h"

#import <pal/spi/cf/CFNetworkSPI.h>
#import <wtf/Function.h>
#import <wtf/RetainPtr.h>
#import <wtf/Vector.h>
#import <unicode/uchar.h>
#import <unicode/uidna.h>
#import <unicode/uscript.h>

namespace WebCore {

NSURL *URLByCanonicalizingURL(NSURL *URL)
{
    RetainPtr<NSURLRequest> request = adoptNS([[NSURLRequest alloc] initWithURL:URL]);
#if HAVE(NSURLPROTOCOL_WITH_SKIPAPPSSO)
    Class concreteClass = [NSURLProtocol _protocolClassForRequest:request.get() skipAppSSO:YES];
#else
    Class concreteClass = [NSURLProtocol _protocolClassForRequest:request.get()];
#endif
    if (!concreteClass) {
        return URL;
    }
    
    // This applies NSURL's concept of canonicalization, but not URL's concept. It would
    // make sense to apply both, but when we tried that it caused a performance degradation
    // (see 5315926). It might make sense to apply only the URL concept and not the NSURL
    // concept, but it's too risky to make that change for WebKit 3.0.
    NSURLRequest *newRequest = [concreteClass canonicalRequestForRequest:request.get()];
    return retainPtr([newRequest URL]).autorelease();
}

} // namespace WebCore
