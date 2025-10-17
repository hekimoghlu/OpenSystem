/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 19, 2022.
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
#import "_WKUserContentFilterInternal.h"

#import "WKContentRuleListInternal.h"
#import "WebCompiledContentRuleList.h"
#import <WebCore/ContentExtensionCompiler.h>
#import <WebCore/ContentExtensionError.h>
#import <string>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-implementations"
@implementation _WKUserContentFilter
#pragma clang diagnostic pop

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return [_contentRuleList _apiObject];
}

@end

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-implementations"
@implementation _WKUserContentFilter (WKPrivate)
#pragma clang diagnostic pop

- (id)_initWithWKContentRuleList:(WKContentRuleList*)contentRuleList
{
    self = [super init];
    if (!self)
        return nil;
    
    _contentRuleList = contentRuleList;
    
    return self;
}

@end
