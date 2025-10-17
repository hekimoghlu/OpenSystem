/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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
#import "WKContentRuleListInternal.h"

#import "WKError.h"
#import "WebCompiledContentRuleList.h"
#import <WebCore/WebCoreObjCExtras.h>

@implementation WKContentRuleList

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKContentRuleList.class, self))
        return;

    _contentRuleList->~ContentRuleList();

    [super dealloc];
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_contentRuleList;
}

- (NSString *)identifier
{
#if ENABLE(CONTENT_EXTENSIONS)
    return _contentRuleList->name();
#else
    return nil;
#endif
}

@end

@implementation WKContentRuleList (WKPrivate)

+ (BOOL)_supportsRegularExpression:(NSString *)regex
{
#if ENABLE(CONTENT_EXTENSIONS)
    return API::ContentRuleList::supportsRegularExpression(regex);
#else
    return NO;
#endif
}

+ (NSError *)_parseRuleList:(NSString *)ruleList
{
#if ENABLE(CONTENT_EXTENSIONS)
    std::error_code error = API::ContentRuleList::parseRuleList(ruleList);
    if (!error)
        return nil;

    auto userInfo = @{ NSHelpAnchorErrorKey: [NSString stringWithFormat:@"Rule list parsing failed: %s", error.message().c_str()] };
    return [NSError errorWithDomain:WKErrorDomain code:WKErrorContentRuleListStoreCompileFailed userInfo:userInfo];
#else
    return nil;
#endif
}

@end
