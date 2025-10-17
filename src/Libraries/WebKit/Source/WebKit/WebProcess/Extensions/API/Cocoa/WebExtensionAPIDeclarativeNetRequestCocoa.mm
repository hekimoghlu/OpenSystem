/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 11, 2025.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "WebExtensionAPIDeclarativeNetRequest.h"

#import "CocoaHelpers.h"
#import "JSWebExtensionWrapper.h"
#import "MessageSenderInlines.h"
#import "WKContentRuleListPrivate.h"
#import "WebExtensionContext.h"
#import "WebExtensionContextMessages.h"
#import "WebExtensionMatchedRuleParameters.h"
#import "WebExtensionUtilities.h"
#import "WebProcess.h"
#import "_WKWebExtensionDeclarativeNetRequestRule.h"
#import <wtf/cocoa/VectorCocoa.h>

#if ENABLE(WK_WEB_EXTENSIONS)

namespace WebKit {

static NSString * const disableRulesetsKey = @"disableRulesetIds";
static NSString * const enableRulesetsKey = @"enableRulesetIds";

static NSString * const regexKey = @"regex";
static NSString * const regexIsCaseSensitiveKey = @"isCaseSensitive";
static NSString * const regexRequireCapturingKey = @"requireCapturing";

static NSString * const actionCountDisplayActionCountAsBadgeTextKey = @"displayActionCountAsBadgeText";
static NSString * const actionCountTabUpdateKey = @"tabUpdate";
static NSString * const actionCountTabIDKey = @"tabId";
static NSString * const actionCountIncrementKey = @"increment";

static NSString * const getMatchedRulesTabIDKey = @"tabId";
static NSString * const getMatchedRulesMinTimeStampKey = @"minTimeStamp";

static NSString * const getDynamicOrSessionRulesRuleIDsKey = @"ruleIds";

static NSString * const addRulesKey = @"addRules";
static NSString * const removeRulesKey = @"removeRuleIds";

void WebExtensionAPIDeclarativeNetRequest::updateEnabledRulesets(NSDictionary *options, Ref<WebExtensionCallbackHandler>&& callback, NSString **outExceptionString)
{
    static NSDictionary<NSString *, id> *types = @{
        disableRulesetsKey: @[ NSString.class ],
        enableRulesetsKey: @[ NSString.class ],
    };

    if (!validateDictionary(options, @"options", nil, types, outExceptionString))
        return;

    Vector<String> rulesetsToEnable = makeVector<String>(objectForKey<NSArray>(options, enableRulesetsKey, true, NSString.class));
    Vector<String> rulesetsToDisable = makeVector<String>(objectForKey<NSArray>(options, disableRulesetsKey, true, NSString.class));

    WebProcess::singleton().sendWithAsyncReply(Messages::WebExtensionContext::DeclarativeNetRequestUpdateEnabledRulesets(rulesetsToEnable, rulesetsToDisable), [protectedThis = Ref { *this }, callback = WTFMove(callback)](Expected<void, WebExtensionError>&& result) {
        if (!result) {
            callback->reportError(result.error());
            return;
        }

        callback->call();
    }, extensionContext().identifier());
}

void WebExtensionAPIDeclarativeNetRequest::getEnabledRulesets(Ref<WebExtensionCallbackHandler>&& callback)
{
    WebProcess::singleton().sendWithAsyncReply(Messages::WebExtensionContext::DeclarativeNetRequestGetEnabledRulesets(), [protectedThis = Ref { *this }, callback = WTFMove(callback)](Vector<String> enabledRulesets) {
        callback->call(createNSArray(enabledRulesets).get());
    }, extensionContext().identifier());
}

void WebExtensionAPIDeclarativeNetRequest::updateDynamicRules(NSDictionary *options, Ref<WebExtensionCallbackHandler>&& callback, NSString **outExceptionString)
{
    static NSDictionary<NSString *, id> *keyTypes = @{
        addRulesKey: @[ NSDictionary.class ],
        removeRulesKey: @[ NSNumber.class ],
    };

    if (!validateDictionary(options, @"options", nil, keyTypes, outExceptionString))
        return;

    auto *rulesToAdd = objectForKey<NSArray>(options, addRulesKey, false, NSDictionary.class);

    NSString *ruleErrorString;
    size_t index = 0;
    for (NSDictionary *ruleDictionary in rulesToAdd) {
        if (![[_WKWebExtensionDeclarativeNetRequestRule  alloc] initWithDictionary:ruleDictionary errorString:&ruleErrorString]) {
            ASSERT(ruleErrorString);
            *outExceptionString = toErrorString(nullString(), addRulesKey, @"an error with rule at index %lu: %@", index, ruleErrorString);
            return;
        }

        ++index;
    }

    String rulesToAddJSON;
    if (rulesToAdd)
        rulesToAddJSON = encodeJSONString(rulesToAdd, JSONOptions::FragmentsAllowed);

    Vector<double> ruleIDsToRemove;
    if (NSArray *rulesToRemove = objectForKey<NSArray>(options, removeRulesKey, false, NSNumber.class)) {
        for (NSNumber *ruleIDToRemove in rulesToRemove)
            ruleIDsToRemove.append(ruleIDToRemove.doubleValue);
    }

    WebProcess::singleton().sendWithAsyncReply(Messages::WebExtensionContext::DeclarativeNetRequestUpdateDynamicRules(WTFMove(rulesToAddJSON), WTFMove(ruleIDsToRemove)), [protectedThis = Ref { *this }, callback = WTFMove(callback)](Expected<void, WebExtensionError>&& result) {
        if (!result) {
            callback->reportError(result.error());
            return;
        }

        callback->call();
    }, extensionContext().identifier());
}

void WebExtensionAPIDeclarativeNetRequest::getDynamicRules(NSDictionary *filter, Ref<WebExtensionCallbackHandler>&& callback)
{
    NSString *outExceptionString;

    static NSDictionary<NSString *, id> *keyTypes = @{
        getDynamicOrSessionRulesRuleIDsKey: @[ NSNumber.class ]
    };

    if (!validateDictionary(filter, nil, nil, keyTypes, &outExceptionString)) {
        callback->reportError(outExceptionString);
        return;
    }

    Vector<double> ruleIDs;
    for (NSNumber *ruleID in filter[getDynamicOrSessionRulesRuleIDsKey])
        ruleIDs.append(ruleID.doubleValue);

    WebProcess::singleton().sendWithAsyncReply(Messages::WebExtensionContext::DeclarativeNetRequestGetDynamicRules(WTFMove(ruleIDs)), [protectedThis = Ref { *this }, callback = WTFMove(callback)](Expected<String, WebExtensionError>&& result) {
        if (!result) {
            callback->reportError(result.error());
            return;
        }

        callback->call(parseJSON(result.value(), JSONOptions::FragmentsAllowed));
    }, extensionContext().identifier());
}

void WebExtensionAPIDeclarativeNetRequest::updateSessionRules(NSDictionary *options, Ref<WebExtensionCallbackHandler>&& callback, NSString **outExceptionString)
{
    static NSDictionary<NSString *, id> *keyTypes = @{
        addRulesKey: @[ NSDictionary.class ],
        removeRulesKey: @[ NSNumber.class ],
    };

    if (!validateDictionary(options, @"options", nil, keyTypes, outExceptionString))
        return;

    auto *rulesToAdd = objectForKey<NSArray>(options, addRulesKey, false, NSDictionary.class);

    NSString *ruleErrorString;
    size_t index = 0;
    for (NSDictionary *ruleDictionary in rulesToAdd) {
        if (![[_WKWebExtensionDeclarativeNetRequestRule  alloc] initWithDictionary:ruleDictionary errorString:&ruleErrorString]) {
            ASSERT(ruleErrorString);
            *outExceptionString = toErrorString(nullString(), addRulesKey, @"an error with rule at index %lu: %@", index, ruleErrorString);
            return;
        }

        ++index;
    }

    String rulesToAddJSON;
    if (rulesToAdd)
        rulesToAddJSON = encodeJSONString(rulesToAdd, JSONOptions::FragmentsAllowed);

    Vector<double> ruleIDsToRemove;
    if (NSArray *rulesToRemove = objectForKey<NSArray>(options, removeRulesKey, false, NSNumber.class)) {
        for (NSNumber *ruleIDToRemove in rulesToRemove)
            ruleIDsToRemove.append(ruleIDToRemove.doubleValue);
    }

    WebProcess::singleton().sendWithAsyncReply(Messages::WebExtensionContext::DeclarativeNetRequestUpdateSessionRules(WTFMove(rulesToAddJSON), WTFMove(ruleIDsToRemove)), [protectedThis = Ref { *this }, callback = WTFMove(callback)](Expected<void, WebExtensionError>&& result) {
        if (!result) {
            callback->reportError(result.error());
            return;
        }

        callback->call();
    }, extensionContext().identifier());
}

void WebExtensionAPIDeclarativeNetRequest::getSessionRules(NSDictionary *filter, Ref<WebExtensionCallbackHandler>&& callback)
{
    NSString *outExceptionString;

    static NSDictionary<NSString *, id> *keyTypes = @{
        getDynamicOrSessionRulesRuleIDsKey: @[ NSNumber.class ]
    };

    if (!validateDictionary(filter, nil, nil, keyTypes, &outExceptionString)) {
        callback->reportError(outExceptionString);
        return;
    }

    Vector<double> ruleIDs;
    for (NSNumber *ruleID in filter[getDynamicOrSessionRulesRuleIDsKey])
        ruleIDs.append(ruleID.doubleValue);

    WebProcess::singleton().sendWithAsyncReply(Messages::WebExtensionContext::DeclarativeNetRequestGetSessionRules(WTFMove(ruleIDs)), [protectedThis = Ref { *this }, callback = WTFMove(callback)](Expected<String, WebExtensionError>&& result) {
        if (!result) {
            callback->reportError(result.error());
            return;
        }

        callback->call(parseJSON(result.value(), JSONOptions::FragmentsAllowed));
    }, extensionContext().identifier());
}

static NSDictionary *toWebAPI(const Vector<WebExtensionMatchedRuleParameters>& matchedRules)
{
    NSMutableArray *matchedRuleArray = [NSMutableArray arrayWithCapacity:matchedRules.size()];

    for (auto& matchedRule : matchedRules) {
        [matchedRuleArray addObject:@{
            @"request": @{ @"url": matchedRule.url.string() },
            @"timeStamp": @(floor(matchedRule.timeStamp.secondsSinceEpoch().milliseconds())),
            @"tabId": @(toWebAPI(matchedRule.tabIdentifier))
        }];
    }

    return @{ @"rulesMatchedInfo": matchedRuleArray };
}

void WebExtensionAPIDeclarativeNetRequest::getMatchedRules(NSDictionary *filter, Ref<WebExtensionCallbackHandler>&& callback, NSString **outExceptionString)
{
    bool hasFeedbackPermission = extensionContext().hasPermission("declarativeNetRequestFeedback"_s);
    bool hasActiveTabPermission = extensionContext().hasPermission("activeTab"_s);

    if (!hasFeedbackPermission && !hasActiveTabPermission) {
        *outExceptionString = toErrorString(nullString(), nullString(), @"either the 'declarativeNetRequestFeedback' or 'activeTab' permission is required");
        return;
    }

    static NSArray<NSString *> *requiredKeysForActiveTab = @[
        getMatchedRulesTabIDKey,
    ];

    static NSDictionary<NSString *, id> *keyTypes = @{
        getMatchedRulesTabIDKey: NSNumber.class,
        getMatchedRulesMinTimeStampKey: NSNumber.class,
    };

    NSArray<NSString *> *requiredKeys = !hasFeedbackPermission ? requiredKeysForActiveTab : @[ ];
    if (!validateDictionary(filter, nil, requiredKeys, keyTypes, outExceptionString))
        return;

    NSNumber *tabID = objectForKey<NSNumber>(filter, getMatchedRulesTabIDKey);
    auto optionalTabIdentifier = tabID ? toWebExtensionTabIdentifier(tabID.doubleValue) : std::nullopt;
    if (tabID && !isValid(optionalTabIdentifier)) {
        *outExceptionString = toErrorString(nullString(), getMatchedRulesTabIDKey, @"%@ is not a valid tab identifier", tabID);
        return;
    }

    NSNumber *minTimeStamp = objectForKey<NSNumber>(filter, getMatchedRulesMinTimeStampKey);
    std::optional<WallTime> optionalTimeStamp;
    if (minTimeStamp)
        optionalTimeStamp = WallTime::fromRawSeconds(Seconds::fromMilliseconds(minTimeStamp.doubleValue).value());

    WebProcess::singleton().sendWithAsyncReply(Messages::WebExtensionContext::DeclarativeNetRequestGetMatchedRules(optionalTabIdentifier, optionalTimeStamp), [protectedThis = Ref { *this }, callback = WTFMove(callback)](Expected<Vector<WebExtensionMatchedRuleParameters>, WebExtensionError>&& result) {
        if (!result) {
            callback->reportError(result.error());
            return;
        }

        callback->call(toWebAPI(result.value()));
    }, extensionContext().identifier());
}

void WebExtensionAPIDeclarativeNetRequest::isRegexSupported(NSDictionary *options, Ref<WebExtensionCallbackHandler>&& callback, NSString **outExceptionString)
{
    static NSDictionary<NSString *, Class> *types = @{
        regexKey: NSString.class,
        regexIsCaseSensitiveKey: @YES.class,
        regexRequireCapturingKey: @YES.class,
    };

    if (!validateDictionary(options, @"regexOptions", @[ regexKey ], types, outExceptionString))
        return;

    NSString *regexString = objectForKey<NSString>(options, regexKey);
    if (![WKContentRuleList _supportsRegularExpression:regexString])
        callback->call(@{ @"isSupported": @NO, @"reason": @"syntaxError" });
    else
        callback->call(@{ @"isSupported": @YES });
}

void WebExtensionAPIDeclarativeNetRequest::setExtensionActionOptions(NSDictionary *options, Ref<WebExtensionCallbackHandler>&& callback, NSString **outExceptionString)
{
    static NSDictionary<NSString *, Class> *types = @{
        actionCountDisplayActionCountAsBadgeTextKey: @YES.class,
        actionCountTabUpdateKey: NSDictionary.class
    };

    if (!validateDictionary(options, @"extensionActionOptions", nil, types, outExceptionString))
        return;

    if (NSDictionary *tabUpdateDictionary = objectForKey<NSDictionary>(options, actionCountTabUpdateKey)) {
        static NSDictionary<NSString *, Class> *tabUpdateTypes = @{
            actionCountTabIDKey: NSNumber.class,
            actionCountIncrementKey: NSNumber.class
        };

        if (!validateDictionary(tabUpdateDictionary, @"tabUpdate", @[ actionCountTabIDKey, actionCountIncrementKey ], tabUpdateTypes, outExceptionString))
            return;

        NSNumber *tabID = objectForKey<NSNumber>(tabUpdateDictionary, actionCountTabIDKey);
        auto tabIdentifier = toWebExtensionTabIdentifier(tabID.doubleValue);
        if (!isValid(tabIdentifier)) {
            *outExceptionString = toErrorString(nullString(), actionCountTabIDKey, @"%@ is not a valid tab identifier", tabID);
            return;
        }

        WebProcess::singleton().sendWithAsyncReply(Messages::WebExtensionContext::DeclarativeNetRequestIncrementActionCount(tabIdentifier.value(), objectForKey<NSNumber>(tabUpdateDictionary, actionCountIncrementKey).doubleValue), [protectedThis = Ref { *this }, callback = WTFMove(callback)](Expected<void, WebExtensionError>&& result) {
            if (!result) {
                callback->reportError(result.error());
                return;
            }

            callback->call();
        }, extensionContext().identifier());
        return;
    }

    WebProcess::singleton().sendWithAsyncReply(Messages::WebExtensionContext::DeclarativeNetRequestDisplayActionCountAsBadgeText(objectForKey<NSNumber>(options, actionCountDisplayActionCountAsBadgeTextKey).boolValue), [protectedThis = Ref { *this }, callback = WTFMove(callback)](Expected<void, WebExtensionError>&& result) {
        if (!result) {
            callback->reportError(result.error());
            return;
        }

        callback->call();
    }, extensionContext().identifier());
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
