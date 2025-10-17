/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 17, 2024.
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
@class WebFrame;
@class WebView;

@interface NSObject (WebMailDelegate)
// Just a simple delegate call like WebFrameLoadDelegate has, but that gets called on the WebThread itself.
- (void)_webthread_webView:(WebView *)sender didFinishLoadForFrame:(WebFrame *)frame;
- (void)_webthread_webView:(WebView *)sender tileDidDraw:(CALayer *)layer;
- (void)_webthread_webView:(WebView *)sender didFinishDocumentLoadForFrame:(WebFrame *)frame;
@end
