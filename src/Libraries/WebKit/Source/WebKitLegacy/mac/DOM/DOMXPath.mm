/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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
#import "DOMInternal.h" // import first to make the private/public trick work
#import "DOMXPath.h"

#import <WebCore/WebScriptObjectPrivate.h>
#import <WebCore/XPathNSResolver.h>
#import <wtf/text/WTFString.h>

//------------------------------------------------------------------------------------------
// DOMNativeXPathNSResolver

@implementation DOMNativeXPathNSResolver

#define IMPL reinterpret_cast<WebCore::XPathNSResolver*>(_internal)

- (void)dealloc
{
    if (_internal)
        IMPL->deref();
    [super dealloc];
}

- (NSString *)lookupNamespaceURI:(NSString *)prefix
{
    return IMPL->lookupNamespaceURI(prefix);
}

@end

WebCore::XPathNSResolver* core(DOMNativeXPathNSResolver *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::XPathNSResolver*>(wrapper->_internal) : 0;
}

DOMNativeXPathNSResolver *kit(WebCore::XPathNSResolver* impl)
{
    if (!impl)
        return nil;
    
    if (DOMNativeXPathNSResolver *wrapper = getDOMWrapper(impl))
        return retainPtr(wrapper).autorelease();
    
    auto wrapper = adoptNS([[DOMNativeXPathNSResolver alloc] _init]);
    wrapper->_internal = reinterpret_cast<DOMObjectInternal*>(impl);
    impl->ref();
    addDOMWrapper(wrapper.get(), impl);
    return wrapper.autorelease();    
}

#undef IMPL
