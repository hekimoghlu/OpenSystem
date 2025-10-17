/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 13, 2023.
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
#import "_WKWebExtensionDeclarativeNetRequestTranslator.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "CocoaHelpers.h"
#import "_WKWebExtensionDeclarativeNetRequestRule.h"

static const NSUInteger maximumNumberOfDeclarativeNetRequestErrorsToSurface = 50;

using namespace WebKit;

@implementation _WKWebExtensionDeclarativeNetRequestTranslator

+ (NSArray<NSDictionary<NSString *, id> *> *)translateRules:(NSArray<NSArray<NSDictionary *> *> *)jsonObjects errorStrings:(NSArray **)outErrorStrings
{
    NSMutableArray<_WKWebExtensionDeclarativeNetRequestRule *> *allValidatedRules = [NSMutableArray array];
    NSMutableArray<NSString *> *errorStrings = [NSMutableArray array];
    NSUInteger totalErrorCount = 0;
    for (NSArray *json in jsonObjects) {
        for (NSDictionary *ruleJSON in json) {
            NSString *errorString;
            _WKWebExtensionDeclarativeNetRequestRule *rule = [[_WKWebExtensionDeclarativeNetRequestRule alloc] initWithDictionary:ruleJSON errorString:&errorString];

            if (rule)
                [allValidatedRules addObject:rule];
            else if (errorString) {
                totalErrorCount++;

                if (errorStrings.count < maximumNumberOfDeclarativeNetRequestErrorsToSurface)
                    [errorStrings addObject:errorString];
            }
        }
    }

    if (totalErrorCount > maximumNumberOfDeclarativeNetRequestErrorsToSurface)
        [errorStrings addObject:@"Error limit hit. No longer omitting errors."];

    if (outErrorStrings)
        *outErrorStrings = [errorStrings copy];

    allValidatedRules = [allValidatedRules sortedArrayUsingComparator:^NSComparisonResult(_WKWebExtensionDeclarativeNetRequestRule *a, _WKWebExtensionDeclarativeNetRequestRule *b) {
        return [a compare:b];
    }].mutableCopy;

    NSMutableArray<NSDictionary<NSString *, id> *> *translatedRules = [NSMutableArray array];
    for (_WKWebExtensionDeclarativeNetRequestRule *rule in allValidatedRules) {
        NSArray<NSDictionary<NSString *, id> *> *translatedRule = rule.ruleInWebKitFormat;
        [translatedRules addObjectsFromArray:translatedRule];
    }

    return translatedRules;
}

+ (NSArray<NSArray<NSDictionary *> *> *)jsonObjectsFromData:(NSArray<NSData *> *)jsonDataArray errorStrings:(NSArray **)outErrorStrings
{
    NSMutableArray *allJSONObjects = [NSMutableArray array];
    NSMutableArray<NSString *> *errors = [NSMutableArray array];
    for (NSData *jsonData in jsonDataArray) {
        NSError *error;
        NSArray<NSDictionary *> *json = parseJSON(jsonData, JSONOptions::FragmentsAllowed, &error);

        // The top level of a declarativeNetRequest ruleset should be an array.
        if (json && [json isKindOfClass:NSArray.class])
            [allJSONObjects addObject:json];

        if (error)
            [errors addObject:error.userInfo[NSDebugDescriptionErrorKey]];
    }

    if (outErrorStrings)
        *outErrorStrings = [errors copy];

    return allJSONObjects;
}

@end

#else

@implementation _WKWebExtensionDeclarativeNetRequestTranslator

+ (NSArray<NSDictionary<NSString *, id> *> *)translateRules:(NSArray<NSArray<NSDictionary *> *> *)jsonObjects errorStrings:(NSArray **)outErrorStrings
{
    return nil;
}

+ (NSArray<NSArray<NSDictionary *> *> *)jsonObjectsFromData:(NSArray<NSData *> *)jsonDataArray errorStrings:(NSArray **)outErrorStrings
{
    return nil;
}

@end

#endif // ENABLE(WK_WEB_EXTENSIONS)
