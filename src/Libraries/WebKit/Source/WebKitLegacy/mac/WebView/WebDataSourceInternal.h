/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 29, 2024.
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
#import "WebDataSourcePrivate.h"
#import <wtf/Forward.h>
#import <wtf/NakedPtr.h>

namespace WebCore {
class DocumentLoader;
class LegacyPreviewLoaderClient;
}

class WebDocumentLoaderMac;

@class DOMDocumentFragment;
@class DOMElement;
@class WebArchive;
@class WebResource;
@class WebView;

@interface WebDataSource (WebInternal)
- (void)_makeRepresentation;
- (BOOL)_isDocumentHTML;
- (WebView *)_webView;
- (NSURL *)_URL;
- (DOMElement *)_imageElementWithImageResource:(WebResource *)resource;
- (DOMDocumentFragment *)_documentFragmentWithImageResource:(WebResource *)resource;
- (DOMDocumentFragment *)_documentFragmentWithArchive:(WebArchive *)archive;
+ (NSMutableDictionary *)_repTypesAllowImageTypeOmission:(BOOL)allowImageTypeOmission;
- (void)_replaceSelectionWithArchive:(WebArchive *)archive selectReplacement:(BOOL)selectReplacement;
- (id)_initWithDocumentLoader:(Ref<WebDocumentLoaderMac>&&)loader;
- (void)_finishedLoading;
- (void)_receivedData:(NSData *)data;
- (void)_revertToProvisionalState;
- (void)_setMainDocumentError:(NSError *)error;
- (NakedPtr<WebCore::DocumentLoader>)_documentLoader;
#if USE(QUICK_LOOK)
@property (nonatomic, copy, setter=_setQuickLookContent:) NSDictionary *_quickLookContent;
@property (nonatomic, setter=_setQuickLookPreviewLoaderClient:) WebCore::LegacyPreviewLoaderClient* _quickLookPreviewLoaderClient;
#endif
@end
