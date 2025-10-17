/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 28, 2024.
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
#import <WebKit/WKWebProcessPlugInBrowserContextController.h>

#import <WebKit/WKBase.h>

@class WKBrowsingContextHandle;
@class _WKRemoteObjectRegistry;
@protocol WKWebProcessPlugInEditingDelegate;
@protocol WKWebProcessPlugInFormDelegatePrivate;

@interface WKWebProcessPlugInBrowserContextController (WKPrivate)

@property (nonatomic, readonly) WKBundlePageRef _bundlePageRef;

@property (nonatomic, readonly) WKBrowsingContextHandle *handle;

@property (nonatomic, readonly) _WKRemoteObjectRegistry *_remoteObjectRegistry;

@property (weak, setter=_setFormDelegate:) id <WKWebProcessPlugInFormDelegatePrivate> _formDelegate;
@property (weak, setter=_setEditingDelegate:) id <WKWebProcessPlugInEditingDelegate> _editingDelegate WK_API_AVAILABLE(macos(10.12.4), ios(10.3));

@property (nonatomic, setter=_setDefersLoading:) BOOL _defersLoading WK_API_DEPRECATED("No longer supported", macos(10.10, 10.15), ios(8.0, 13.0));

@property (nonatomic, readonly) BOOL _usesNonPersistentWebsiteDataStore;

@property (nonatomic, readonly) NSString *_groupIdentifier WK_API_AVAILABLE(macos(12.0), ios(15.0));

+ (instancetype)lookUpBrowsingContextFromHandle:(WKBrowsingContextHandle *)handle;

@end
