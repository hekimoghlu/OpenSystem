/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 29, 2024.
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
#import "PreviewConverter.h"

#if ENABLE(PREVIEW_CONVERTER) && USE(QUICK_LOOK)

#import "PreviewConverterClient.h"
#import "PreviewConverterProvider.h"
#import "QuickLook.h"
#import "ResourceError.h"
#import "ResourceRequest.h"
#import "ResourceResponse.h"
#import "SharedBuffer.h"
#import <pal/ios/QuickLookSoftLink.h>

@interface WebPreviewConverterDelegate : NSObject
- (instancetype)initWithDelegate:(WebCore::PreviewPlatformDelegate&)delegate;
@end

@implementation WebPreviewConverterDelegate {
    WeakPtr<WebCore::PreviewPlatformDelegate> _delegate;
}

- (instancetype)initWithDelegate:(WebCore::PreviewPlatformDelegate&)delegate
{
    if (!(self = [super init]))
        return nil;

    _delegate = delegate;
    return self;
}

- (void)connection:(NSURLConnection *)connection didReceiveData:(NSData *)data lengthReceived:(long long)lengthReceived
{
    ASSERT_UNUSED(connection, !connection);
    ASSERT_UNUSED(lengthReceived, lengthReceived >= 0);
    ASSERT(data.length == static_cast<NSUInteger>(lengthReceived));
    if (auto delegate = _delegate.get())
        delegate->delegateDidReceiveData(WebCore::SharedBuffer::create(data).get());
}

- (void)connectionDidFinishLoading:(NSURLConnection *)connection
{
    ASSERT_UNUSED(connection, !connection);
    if (auto delegate = _delegate.get())
        delegate->delegateDidFinishLoading();
}

- (void)connection:(NSURLConnection *)connection didFailWithError:(NSError *)error
{
    ASSERT_UNUSED(connection, !connection);
    if (auto delegate = _delegate.get())
        delegate->delegateDidFailWithError(error);
}

@end

namespace WebCore {

PreviewConverter::PreviewConverter(const ResourceResponse& response, PreviewConverterProvider& provider)
    : m_previewData { FragmentedSharedBuffer::create() }
    , m_originalResponse { response }
    , m_provider { provider }
    , m_platformDelegate { adoptNS([[WebPreviewConverterDelegate alloc] initWithDelegate:*this]) }
    , m_platformConverter { adoptNS([PAL::allocQLPreviewConverterInstance() initWithConnection:nil delegate:m_platformDelegate.get() response:m_originalResponse.nsURLResponse() options:nil]) }
{
}

UncheckedKeyHashSet<String, ASCIICaseInsensitiveHash> PreviewConverter::platformSupportedMIMETypes()
{
    UncheckedKeyHashSet<String, ASCIICaseInsensitiveHash> supportedMIMETypes;
    for (NSString *mimeType in QLPreviewGetSupportedMIMETypesSet())
        supportedMIMETypes.add(mimeType);
    return supportedMIMETypes;
}

ResourceRequest PreviewConverter::safeRequest(const ResourceRequest& request) const
{
    return [m_platformConverter safeRequestForRequest:request.nsURLRequest(HTTPBodyUpdatePolicy::DoNotUpdateHTTPBody)];
}

ResourceResponse PreviewConverter::platformPreviewResponse() const
{
    ResourceResponse response { [m_platformConverter previewResponse] };
    ASSERT(response.url().protocolIs(QLPreviewProtocol));
    return response;
}

String PreviewConverter::previewFileName() const
{
    return [m_platformConverter previewFileName];
}

String PreviewConverter::previewUTI() const
{
    return [m_platformConverter previewUTI];
}

void PreviewConverter::platformAppend(const SharedBufferDataView& data)
{
    [m_platformConverter appendData:data.createNSData().get()];
}

void PreviewConverter::platformFinishedAppending()
{
    [m_platformConverter finishedAppendingData];
}

void PreviewConverter::platformFailedAppending()
{
    [m_platformConverter finishConverting];
}

bool PreviewConverter::isPlatformPasswordError(const ResourceError& error) const
{
    return error.errorCode() == kQLReturnPasswordProtected && error.domain() == "QuickLookErrorDomain"_s;
}

static NSDictionary *optionsWithPassword(const String& password)
{
    if (password.isNull())
        return nil;
    
    return @{ (NSString *)PAL::get_QuickLook_kQLPreviewOptionPasswordKey() : password };
}

void PreviewConverter::platformUnlockWithPassword(const String& password)
{
    m_platformConverter = adoptNS([PAL::allocQLPreviewConverterInstance() initWithConnection:nil delegate:m_platformDelegate.get() response:m_originalResponse.nsURLResponse() options:optionsWithPassword(password)]);
}

} // namespace WebCore

#endif // ENABLE(PREVIEW_CONVERTER) && USE(QUICK_LOOK)
