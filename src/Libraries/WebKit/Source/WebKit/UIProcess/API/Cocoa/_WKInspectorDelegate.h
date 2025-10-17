/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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
#import <Foundation/Foundation.h>
#import <WebKit/WKFoundation.h>

@class _WKInspector;

@protocol _WKInspectorDelegate <NSObject>
@optional

/*! @abstract Called when the _WKInspector requests to show a resource externally. This
    is used to display documentation pages and to show external URLs that are linkified.
    @param inspector the associated inspector for which an external navigation should be triggered.
    @param url The resource to be shown.
 */
- (void)inspector:(_WKInspector *)inspector openURLExternally:(NSURL *)url;

/*! @abstract Called when the _WKInspector user interface has been fully loaded.
    @param inspector the associated inspector that has finished loading.
*/
- (void)inspectorFrontendLoaded:(_WKInspector *)inspector;

@end
