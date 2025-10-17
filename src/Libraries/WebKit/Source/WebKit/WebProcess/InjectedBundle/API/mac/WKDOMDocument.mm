/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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
#import "WKDOMDocument.h"

#import "WKDOMInternals.h"
#import <WebCore/Document.h>
#import <WebCore/DocumentFragment.h>
#import <WebCore/HTMLElement.h>
#import <WebCore/SimpleRange.h>
#import <WebCore/Text.h>
#import <WebCore/markup.h>
#import <wtf/NakedRef.h>

@interface WKDOMDocumentParserYieldToken : NSObject

@end

@implementation WKDOMDocumentParserYieldToken {
    std::unique_ptr<WebCore::DocumentParserYieldToken> _token;
}

- (instancetype)initWithDocument:(NakedRef<WebCore::Document>)document
{
    if (self = [super init])
        _token = document->createParserYieldToken();
    return self;
}

@end

@implementation WKDOMDocument

- (WKDOMElement *)createElement:(NSString *)tagName
{
    // FIXME: Do something about the exception.
    auto result = downcast<WebCore::Document>(*_impl).createElementForBindings(tagName);
    if (result.hasException())
        return nil;
    return WebKit::toWKDOMElement(result.releaseReturnValue().ptr());
}

- (WKDOMText *)createTextNode:(NSString *)data
{
    return WebKit::toWKDOMText(downcast<WebCore::Document>(*_impl).createTextNode(data).ptr());
}

- (WKDOMElement *)body
{
    return WebKit::toWKDOMElement(downcast<WebCore::Document>(*_impl).bodyOrFrameset());
}

- (WKDOMNode *)createDocumentFragmentWithMarkupString:(NSString *)markupString baseURL:(NSURL *)baseURL
{
    return WebKit::toWKDOMNode(createFragmentFromMarkup(downcast<WebCore::Document>(*_impl), markupString, baseURL.absoluteString).ptr());
}

- (WKDOMNode *)createDocumentFragmentWithText:(NSString *)text
{
    return WebKit::toWKDOMNode(createFragmentFromText(makeRangeSelectingNodeContents(downcast<WebCore::Document>(*_impl)), text).ptr());
}

- (id)parserYieldToken
{
    return adoptNS([[WKDOMDocumentParserYieldToken alloc] initWithDocument:downcast<WebCore::Document>(*_impl)]).autorelease();
}

@end
