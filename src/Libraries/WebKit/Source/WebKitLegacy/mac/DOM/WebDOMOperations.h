/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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
#import <WebKitLegacy/DOMCore.h>
#import <WebKitLegacy/DOMHTML.h>
#import <WebKitLegacy/DOMRange.h>

@class WebArchive;
@class WebFrame;

@interface DOMNode (WebDOMNodeOperations)

/*!
    @property webArchive
    @abstract A WebArchive representing the node and the children of the node.
*/
@property (nonatomic, readonly, strong) WebArchive *webArchive;

@end

@interface DOMDocument (WebDOMDocumentOperations)

/*!
    @property webFrame
    @abstract The frame of the DOM document.
*/
@property (nonatomic, readonly, strong) WebFrame *webFrame;

/*!
    @method URLWithAttributeString:
    @abstract Constructs a URL given an attribute string.
    @discussion This method constructs a URL given an attribute string just as WebKit does. 
    An attribute string is the value of an attribute of an element such as the href attribute on 
    the DOMHTMLAnchorElement class. This method is only applicable to attributes that refer to URLs.
*/
- (NSURL *)URLWithAttributeString:(NSString *)string;

@end

@interface DOMRange (WebDOMRangeOperations)

/*!
    @property webArchive
    @abstract A WebArchive representing the range.
*/
@property (nonatomic, readonly, strong) WebArchive *webArchive;

/*!
    @property markupString
    @abstract A markup string representing the range.
*/
@property (nonatomic, readonly, copy) NSString *markupString;

@end

@interface DOMHTMLFrameElement (WebDOMHTMLFrameElementOperations)

/*!
    @property contentFrame
    @abstract The content frame of the element.
*/
@property (nonatomic, readonly, strong) WebFrame *contentFrame;

@end

@interface DOMHTMLIFrameElement (WebDOMHTMLIFrameElementOperations)

/*!
    @property contentFrame
    @abstract Returns the content frame of the element.
*/
@property (nonatomic, readonly, strong) WebFrame *contentFrame;

@end

@interface DOMHTMLObjectElement (WebDOMHTMLObjectElementOperations)

/*!
    @property contentFrame
    @abstract The content frame of the element.
    @discussion Returns non-nil only if the object represents a child frame
    such as if the data of the object is HTML content.
*/
@property (nonatomic, readonly, strong) WebFrame *contentFrame;

@end
