/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 29, 2025.
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
#pragma once

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>

typedef NS_ENUM(NSInteger, WKTextExtractionContainer) {
    WKTextExtractionContainerRoot,
    WKTextExtractionContainerViewportConstrained,
    WKTextExtractionContainerList,
    WKTextExtractionContainerListItem,
    WKTextExtractionContainerBlockQuote,
    WKTextExtractionContainerArticle,
    WKTextExtractionContainerSection,
    WKTextExtractionContainerNav,
    WKTextExtractionContainerButton
};

@interface WKTextExtractionItem : NSObject
@property (nonatomic, readonly) NSArray<WKTextExtractionItem *> *children;
@property (nonatomic, readonly) CGRect rectInWebView;
@end

@interface WKTextExtractionContainerItem : WKTextExtractionItem
- (instancetype)initWithContainer:(WKTextExtractionContainer)container rectInWebView:(CGRect)rectInWebView children:(NSArray<WKTextExtractionItem *> *)children;
@property (nonatomic, readonly) WKTextExtractionContainer container;
@end

@interface WKTextExtractionLink : NSObject
- (instancetype)initWithURL:(NSURL *)url range:(NSRange)range;
@property (nonatomic, readonly) NSURL *url;
@property (nonatomic, readonly) NSRange range;
@end

@interface WKTextExtractionEditable : NSObject
- (instancetype)initWithLabel:(NSString *)label placeholder:(NSString *)placeholder isSecure:(BOOL)isSecure isFocused:(BOOL)isFocused;
@property (nonatomic, readonly) NSString *label;
@property (nonatomic, readonly) NSString *placeholder;
@property (nonatomic, readonly, getter=isSecure) BOOL secure;
@property (nonatomic, readonly, getter=isFocused) BOOL focused;
@end

@interface WKTextExtractionTextItem : WKTextExtractionItem
- (instancetype)initWithContent:(NSString *)content selectedRange:(NSRange)selectedRange links:(NSArray<WKTextExtractionLink *> *)links editable:(WKTextExtractionEditable *)editable rectInWebView:(CGRect)rectInWebView children:(NSArray<WKTextExtractionItem *> *)children;
@property (nonatomic, readonly) NSString *content;
@property (nonatomic, readonly) NSRange selectedRange;
@property (nonatomic, readonly) NSArray<WKTextExtractionLink *> *links;
@property (nonatomic, readonly) WKTextExtractionEditable *editable;
@end

@interface WKTextExtractionScrollableItem : WKTextExtractionItem
- (instancetype)initWithContentSize:(CGSize)contentSize rectInWebView:(CGRect)rectInWebView children:(NSArray<WKTextExtractionItem *> *)children;
@property (nonatomic, readonly) CGSize contentSize;
@end

@interface WKTextExtractionImageItem : WKTextExtractionItem
- (instancetype)initWithName:(NSString *)name altText:(NSString *)altText rectInWebView:(CGRect)rectInWebView children:(NSArray<WKTextExtractionItem *> *)children;
@property (nonatomic, readonly) NSString *name;
@property (nonatomic, readonly) NSString *altText;
@end

@interface WKTextExtractionRequest : NSObject
- (void)fulfill:(WKTextExtractionItem *)result;
@property (nonatomic, readonly) CGRect rectInWebView;
@end
