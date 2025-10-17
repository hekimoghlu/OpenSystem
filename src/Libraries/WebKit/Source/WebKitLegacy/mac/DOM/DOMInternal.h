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
#import "DOMNodeFilter.h"
#import "DOMObject.h"
#import "DOMXPathNSResolver.h"
#import <wtf/Forward.h>
#import <wtf/WallTime.h>

namespace JSC {
    class JSObject;
    namespace Bindings {
        class RootObject;
    }
}

namespace WebCore {
class NodeFilter;
    class XPathNSResolver;
#if ENABLE(TOUCH_EVENTS)
    class Touch;
#endif
}

@interface DOMNodeFilter : DOMObject <DOMNodeFilter>
@end

@interface DOMNativeXPathNSResolver : DOMObject <DOMXPathNSResolver>
@end

// Helper functions for DOM wrappers and gluing to Objective-C

void initializeDOMWrapperHooks();

NSObject* getDOMWrapper(DOMObjectInternal*);
void addDOMWrapper(NSObject* wrapper, DOMObjectInternal*);
void removeDOMWrapper(DOMObjectInternal*);

template <class Source>
inline id getDOMWrapper(Source impl)
{
    return getDOMWrapper(reinterpret_cast<DOMObjectInternal*>(impl));
}

template <class Source>
inline void addDOMWrapper(NSObject* wrapper, Source impl)
{
    addDOMWrapper(wrapper, reinterpret_cast<DOMObjectInternal*>(impl));
}

DOMNodeFilter *kit(WebCore::NodeFilter*);
WebCore::NodeFilter* core(DOMNodeFilter *);

DOMNativeXPathNSResolver *kit(WebCore::XPathNSResolver*);
WebCore::XPathNSResolver* core(DOMNativeXPathNSResolver *);

inline NSTimeInterval kit(WallTime time)
{
    return time.secondsSinceEpoch().value() - NSTimeIntervalSince1970;
}

inline WallTime core(NSTimeInterval sec)
{
    return WallTime::fromRawSeconds(sec + NSTimeIntervalSince1970);
}
