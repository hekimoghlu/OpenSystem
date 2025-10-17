/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 7, 2022.
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
#import <WebKit/WKBrowsingContextController.h>

#import <WebKit/WKBase.h>

typedef NS_ENUM(NSUInteger, WKBrowsingContextPaginationMode) {
    WKPaginationModeUnpaginated,
    WKPaginationModeLeftToRight,
    WKPaginationModeRightToLeft,
    WKPaginationModeTopToBottom,
    WKPaginationModeBottomToTop,
};

@class WKBrowsingContextHandle;
@class _WKRemoteObjectRegistry;

@interface WKBrowsingContextController (Private)

@property (readonly) WKPageRef _pageRef;

@property (readonly) BOOL hasOnlySecureContent;

@property WKBrowsingContextPaginationMode paginationMode;

// Whether the column-break-{before,after} properties are respected instead of the
// page-break-{before,after} properties.
@property BOOL paginationBehavesLikeColumns;

// Set to 0 to have the page length equal the view length.
@property CGFloat pageLength;
@property CGFloat gapBetweenPages;

// Whether or not to enable a line grid by default on the paginated content.
@property BOOL paginationLineGridEnabled;

@property (readonly) NSUInteger pageCount;

@property (nonatomic, readonly) WKBrowsingContextHandle *handle;

@property (nonatomic, readonly) _WKRemoteObjectRegistry *_remoteObjectRegistry;

@property (nonatomic, readonly) pid_t processIdentifier;

@end
