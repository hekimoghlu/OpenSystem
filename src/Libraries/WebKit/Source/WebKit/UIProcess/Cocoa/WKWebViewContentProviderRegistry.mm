/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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
#import "WKWebViewContentProviderRegistry.h"

#if PLATFORM(IOS_FAMILY)

#import "WKPDFView.h"
#import "WKPreferencesInternal.h"
#import "WKUSDPreviewView.h"
#import "WKWebViewInternal.h"
#import "WebPageProxy.h"
#import <WebCore/MIMETypeRegistry.h>
#import <WebKit/WKPreferencesPrivate.h>
#import <WebKit/WKWebViewConfigurationPrivate.h>
#import <wtf/FixedVector.h>
#import <wtf/HashCountedSet.h>
#import <wtf/HashMap.h>
#import <wtf/text/StringHash.h>

@implementation WKWebViewContentProviderRegistry {
    HashMap<String, Class <WKWebViewContentProvider>, ASCIICaseInsensitiveHash> _contentProviderForMIMEType;
}

- (instancetype)initWithConfiguration:(WKWebViewConfiguration *)configuration
{
    if (!(self = [super init]))
        return nil;

#if ENABLE(WKPDFVIEW)
    if ([WKPDFView platformSupportsPDFView] && (!configuration.preferences || !configuration.preferences->_preferences->unifiedPDFEnabled())) {
        for (auto& type : WebCore::MIMETypeRegistry::pdfMIMETypes())
            [self registerProvider:[WKPDFView class] forMIMEType:@(type.characters())];
    }
#endif

#if USE(SYSTEM_PREVIEW)
    if (configuration._systemPreviewEnabled && !configuration.preferences._modelDocumentEnabled) {
        for (auto& type : WebCore::MIMETypeRegistry::usdMIMETypes())
            [self registerProvider:[WKUSDPreviewView class] forMIMEType:@(type.characters())];
    }
#endif

    return self;
}

- (void)registerProvider:(Class <WKWebViewContentProvider>)contentProvider forMIMEType:(const String&)mimeType
{
    _contentProviderForMIMEType.set(mimeType, contentProvider);
}

- (Class <WKWebViewContentProvider>)providerForMIMEType:(const String&)mimeType
{
    if (mimeType.isEmpty())
        return nil;

    const auto& representation = _contentProviderForMIMEType.find(mimeType);

    if (representation == _contentProviderForMIMEType.end())
        return nil;

    return representation->value;
}

- (Vector<String>)_mimeTypesWithCustomContentProviders
{
    return copyToVector(_contentProviderForMIMEType.keys());
}

@end

#endif // PLATFORM(IOS_FAMILY)
