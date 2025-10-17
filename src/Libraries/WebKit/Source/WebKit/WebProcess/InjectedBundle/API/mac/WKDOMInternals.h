/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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
#import <WebCore/Node.h>
#import <WebCore/Range.h>
#import <WebKit/WKDOMNode.h>
#import <WebKit/WKDOMRange.h>
#import <wtf/HashMap.h>

namespace WebCore {
class Element;
class Document;
}

@class WKDOMElement;
@class WKDOMDocument;
@class WKDOMText;

@interface WKDOMNode () {
@package
    RefPtr<WebCore::Node> _impl;
}

- (id)_initWithImpl:(WebCore::Node*)impl;
@end

@interface WKDOMRange () {
@package
    RefPtr<WebCore::Range> _impl;
}

- (id)_initWithImpl:(WebCore::Range*)impl;
@end

namespace WebKit {

template<typename WebCoreType, typename WKDOMType>
class DOMCache {
public:
    DOMCache()
    {
    }

    void add(WebCoreType core, WKDOMType kit)
    {
        m_map.add(core, kit);
    }
    
    WKDOMType get(WebCoreType core)
    {
        return m_map.get(core);
    }

    void remove(WebCoreType core)
    {
        m_map.remove(core);
    }

private:
    // This class should only ever be used as a singleton.
    ~DOMCache() = delete;

    HashMap<WebCoreType, WKDOMType> m_map;
};

// -- Caches --

DOMCache<WebCore::Node*, __unsafe_unretained WKDOMNode *>& WKDOMNodeCache();
DOMCache<WebCore::Range*, __unsafe_unretained WKDOMRange *>& WKDOMRangeCache();

// -- Node and classes derived from Node. --

WebCore::Node* toWebCoreNode(WKDOMNode *);
WKDOMNode *toWKDOMNode(WebCore::Node*);

WebCore::Element* toWebCoreElement(WKDOMElement *);
WKDOMElement *toWKDOMElement(WebCore::Element*);

WebCore::Document* toWebCoreDocument(WKDOMDocument *);
WKDOMDocument *toWKDOMDocument(WebCore::Document*);

WebCore::Text* toWebCoreText(WKDOMText *);
WKDOMText *toWKDOMText(WebCore::Text*);

// -- Range. --

WebCore::Range* toWebCoreRange(WKDOMRange *);
WKDOMRange *toWKDOMRange(WebCore::Range*);

} // namespace WebKit
