/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 9, 2024.
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
#import "WKInspectorResourceURLSchemeHandler.h"

#if PLATFORM(MAC)

#import "Logging.h"
#import "WKURLSchemeTask.h"
#import "WebInspectorUIProxy.h"
#import "WebURLSchemeHandlerCocoa.h"
#import <WebCore/MIMETypeRegistry.h>
#import <wtf/Assertions.h>

@implementation WKInspectorResourceURLSchemeHandler {
    RetainPtr<NSMapTable<id <WKURLSchemeTask>, NSOperation *>> _fileLoadOperations;
    RetainPtr<NSBundle> _cachedBundle;
    
    RetainPtr<NSSet<NSString *>> _allowedURLSchemesForCSP;
    RetainPtr<NSSet<NSURL *>> _mainResourceURLsForCSP;
}

- (NSSet<NSString *> *)allowedURLSchemesForCSP
{
    return _allowedURLSchemesForCSP.get();
}

- (void)setAllowedURLSchemesForCSP:(NSSet<NSString *> *)allowedURLSchemes
{
    _allowedURLSchemesForCSP = adoptNS([allowedURLSchemes copy]);
}

- (NSSet<NSURL *> *)mainResourceURLsForCSP
{
    if (!_mainResourceURLsForCSP)
        _mainResourceURLsForCSP = adoptNS([[NSSet alloc] initWithObjects:[NSURL URLWithString:WebKit::WebInspectorUIProxy::inspectorPageURL()], [NSURL URLWithString:WebKit::WebInspectorUIProxy::inspectorTestPageURL()], nil]);

    return _mainResourceURLsForCSP.get();
}

// MARK - WKURLSchemeHandler Protocol

- (void)webView:(WKWebView *)webView startURLSchemeTask:(id <WKURLSchemeTask>)urlSchemeTask
{
    dispatch_assert_queue(dispatch_get_main_queue());
    if (!_cachedBundle) {
        _cachedBundle = [NSBundle bundleWithIdentifier:@"com.apple.WebInspectorUI"];

        // It is an error if WebInspectorUI has not already been soft-linked by the time
        // we load resources from it. And if soft-linking fails, we shouldn't start loads.
        RELEASE_ASSERT(_cachedBundle);
    }

    if (!_fileLoadOperations)
        _fileLoadOperations = adoptNS([[NSMapTable alloc] initWithKeyOptions:NSPointerFunctionsStrongMemory valueOptions:NSPointerFunctionsStrongMemory capacity:5]);

    NSBlockOperation *operation = [NSBlockOperation blockOperationWithBlock:^{
        [_fileLoadOperations removeObjectForKey:urlSchemeTask];

        NSURL *requestURL = urlSchemeTask.request.URL;
        NSURL *fileURLForRequest = [_cachedBundle URLForResource:requestURL.relativePath withExtension:@""];
        if (!fileURLForRequest) {
            LOG_ERROR("Unable to find Web Inspector resource: %@", requestURL.absoluteString);
            [urlSchemeTask didFailWithError:[NSError errorWithDomain:NSCocoaErrorDomain code:NSURLErrorFileDoesNotExist userInfo:nil]];
            return;
        }

        NSError *readError;
        NSData *fileData = [NSData dataWithContentsOfURL:fileURLForRequest options:0 error:&readError];
        if (!fileData) {
            LOG_ERROR("Unable to read data for Web Inspector resource: %@", requestURL.absoluteString);
            [urlSchemeTask didFailWithError:[NSError errorWithDomain:NSCocoaErrorDomain code:NSURLErrorResourceUnavailable userInfo:@{
                NSUnderlyingErrorKey: readError,
            }]];
            return;
        }

        NSString *mimeType = WebCore::MIMETypeRegistry::mimeTypeForExtension(String(fileURLForRequest.pathExtension));
        if (!mimeType)
            mimeType = @"application/octet-stream";

        RetainPtr<NSMutableDictionary> headerFields = adoptNS(@{
            @"Access-Control-Allow-Origin": @"*",
            @"Content-Length": [NSString stringWithFormat:@"%zu", (size_t)fileData.length],
            @"Content-Type": mimeType,
        }.mutableCopy);

        // Allow fetches for resources that use a registered custom URL scheme.
        if (_allowedURLSchemesForCSP && [self.mainResourceURLsForCSP containsObject:requestURL]) {
            NSString *listOfCustomProtocols = [NSString stringWithFormat:@"%@:", [_allowedURLSchemesForCSP.get().allObjects componentsJoinedByString:@": "]];
            NSString *stringForCSPPolicy = [NSString stringWithFormat:@"connect-src * %@; img-src * file: blob: resource: %@", listOfCustomProtocols, listOfCustomProtocols];
            [headerFields setObject:stringForCSPPolicy forKey:@"Content-Security-Policy"];
        }

        RetainPtr<NSHTTPURLResponse> urlResponse = adoptNS([[NSHTTPURLResponse alloc] initWithURL:urlSchemeTask.request.URL statusCode:200 HTTPVersion:nil headerFields:headerFields.get()]);
        [urlSchemeTask didReceiveResponse:urlResponse.get()];
        [urlSchemeTask didReceiveData:fileData];
        [urlSchemeTask didFinish];
    }];
    
    [_fileLoadOperations setObject:operation forKey:urlSchemeTask];
    [[NSOperationQueue mainQueue] addOperation:operation];
}

- (void)webView:(WKWebView *)webView stopURLSchemeTask:(id <WKURLSchemeTask>)urlSchemeTask
{
    dispatch_assert_queue(dispatch_get_main_queue());
    if (NSOperation *operation = [_fileLoadOperations objectForKey:urlSchemeTask]) {
        [operation cancel];
        [_fileLoadOperations removeObjectForKey:urlSchemeTask];
    }
}

@end

#endif // PLATFORM(MAC)
