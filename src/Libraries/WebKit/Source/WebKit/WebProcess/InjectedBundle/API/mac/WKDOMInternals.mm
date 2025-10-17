/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
#import "WKDOMInternals.h"

#import <WebCore/Document.h>
#import <WebCore/Element.h>
#import <WebCore/Node.h>
#import <WebCore/Range.h>
#import <WebCore/Text.h>
#import <wtf/NeverDestroyed.h>

// Classes to instantiate.
#import "WKDOMElement.h"
#import "WKDOMDocument.h"
#import "WKDOMText.h"

#if PLATFORM(IOS_FAMILY)
#import <WebCore/WAKAppKitStubs.h>
#endif

namespace WebKit {

template<typename WebCoreType, typename WKDOMType>
static WKDOMType toWKDOMType(WebCoreType impl, DOMCache<WebCoreType, WKDOMType>& cache);

// -- Caches -- 

DOMCache<WebCore::Node*, __unsafe_unretained WKDOMNode *>& WKDOMNodeCache()
{
    static NeverDestroyed<DOMCache<WebCore::Node*, __unsafe_unretained WKDOMNode *>> cache;
    return cache;
}

DOMCache<WebCore::Range*, __unsafe_unretained WKDOMRange *>& WKDOMRangeCache()
{
    static NeverDestroyed<DOMCache<WebCore::Range*, __unsafe_unretained WKDOMRange *>> cache;
    return cache;
}

// -- Node and classes derived from Node. --

static Class WKDOMNodeClass(WebCore::Node* impl)
{
    switch (impl->nodeType()) {
    case WebCore::Node::ELEMENT_NODE:
        return [WKDOMElement class];
    case WebCore::Node::DOCUMENT_NODE:
        return [WKDOMDocument class];
    case WebCore::Node::TEXT_NODE:
        return [WKDOMText class];
    case WebCore::Node::ATTRIBUTE_NODE:
    case WebCore::Node::CDATA_SECTION_NODE:
    case WebCore::Node::PROCESSING_INSTRUCTION_NODE:
    case WebCore::Node::COMMENT_NODE:
    case WebCore::Node::DOCUMENT_TYPE_NODE:
    case WebCore::Node::DOCUMENT_FRAGMENT_NODE:
        return [WKDOMNode class];
    }
    ASSERT_NOT_REACHED();
    return nil;
}

static RetainPtr<WKDOMNode> createWrapper(WebCore::Node* impl)
{
    return adoptNS([[WKDOMNodeClass(impl) alloc] _initWithImpl:impl]);
}

WebCore::Node* toWebCoreNode(WKDOMNode *wrapper)
{
    return wrapper ? wrapper->_impl.get() : 0;
}

WKDOMNode *toWKDOMNode(WebCore::Node* impl)
{
    return toWKDOMType<WebCore::Node*, __unsafe_unretained WKDOMNode *>(impl, WKDOMNodeCache());
}

WebCore::Element* toWebCoreElement(WKDOMElement *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::Element*>(wrapper->_impl.get()) : 0;
}

WKDOMElement *toWKDOMElement(WebCore::Element* impl)
{
    return static_cast<WKDOMElement*>(toWKDOMNode(static_cast<WebCore::Node*>(impl)));
}

WebCore::Document* toWebCoreDocument(WKDOMDocument *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::Document*>(wrapper->_impl.get()) : 0;
}

WKDOMDocument *toWKDOMDocument(WebCore::Document* impl)
{
    return static_cast<WKDOMDocument*>(toWKDOMNode(static_cast<WebCore::Node*>(impl)));
}

WebCore::Text* toWebCoreText(WKDOMText *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::Text*>(wrapper->_impl.get()) : 0;
}

WKDOMText *toWKDOMText(WebCore::Text* impl)
{
    return static_cast<WKDOMText*>(toWKDOMNode(static_cast<WebCore::Node*>(impl)));
}

// -- Range. --

static RetainPtr<WKDOMRange> createWrapper(WebCore::Range* impl)
{
    return adoptNS([[WKDOMRange alloc] _initWithImpl:impl]);
}

WebCore::Range* toWebCoreRange(WKDOMRange * wrapper)
{
    return wrapper ? wrapper->_impl.get() : 0;
}

WKDOMRange *toWKDOMRange(WebCore::Range* impl)
{
    return toWKDOMType<WebCore::Range*, __unsafe_unretained WKDOMRange *>(impl, WKDOMRangeCache());
}

// -- Helpers --

template<typename WebCoreType, typename WKDOMType>
static WKDOMType toWKDOMType(WebCoreType impl, DOMCache<WebCoreType, WKDOMType>& cache)
{
    if (!impl)
        return nil;
    if (WKDOMType wrapper = cache.get(impl))
        return retainPtr(wrapper).autorelease();
    auto wrapper = createWrapper(impl);
    if (!wrapper)
        return nil;
    return wrapper.autorelease();
}

} // namespace WebKit
