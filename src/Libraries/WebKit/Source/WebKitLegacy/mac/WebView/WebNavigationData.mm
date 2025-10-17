/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#import "WebNavigationData.h"
#import <wtf/RetainPtr.h>

@interface WebNavigationDataPrivate : NSObject
{
@public
    RetainPtr<NSString> url;
    RetainPtr<NSString> title;
    RetainPtr<NSURLRequest> originalRequest;
    RetainPtr<NSURLResponse> response;
    BOOL hasSubstituteData;
    RetainPtr<NSString> clientRedirectSource;
}

@end

@implementation WebNavigationDataPrivate

@end

@implementation WebNavigationData

- (id)initWithURLString:(NSString *)url title:(NSString *)title originalRequest:(NSURLRequest *)request response:(NSURLResponse *)response hasSubstituteData:(BOOL)hasSubstituteData clientRedirectSource:(NSString *)redirectSource
{
    self = [super init];
    if (!self)
        return nil;
    _private = [[WebNavigationDataPrivate alloc] init];
    
    _private->url = url;
    _private->title = title;
    _private->originalRequest = request;
    _private->response = response;
    _private->hasSubstituteData = hasSubstituteData;
    _private->clientRedirectSource = redirectSource;
    
    return self;
}

- (NSString *)url
{
    return _private->url.get();
}

- (NSString *)title
{
    return _private->title.get();
}

- (NSURLRequest *)originalRequest
{
    return _private->originalRequest.get();
}

- (NSURLResponse *)response
{
    return _private->response.get();
}

- (BOOL)hasSubstituteData
{
    return _private->hasSubstituteData;
}

- (NSString *)clientRedirectSource
{
    return _private->clientRedirectSource.get();
}

- (void)dealloc
{
    [_private release];
    [super dealloc];
}

@end
