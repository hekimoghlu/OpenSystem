/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 19, 2021.
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
#pragma once

#if ENABLE(WK_WEB_EXTENSIONS)

#include "WebExtensionError.h"
#include <JavaScriptCore/JSBase.h>
#include <wtf/Function.h>
#include <wtf/JSONValues.h>
#include <wtf/Markable.h>
#include <wtf/UUID.h>
#include <wtf/Vector.h>

#ifdef __OBJC__
#import <wtf/RetainPtr.h>
#import <wtf/cocoa/VectorCocoa.h>
#endif

namespace WebKit {

class WebFrame;

Ref<JSON::Array> filterObjects(const JSON::Array&, WTF::Function<bool(const JSON::Value&)>&& lambda);

Vector<String> makeStringVector(const JSON::Array&);

Vector<double> availableScreenScales();
double largestDisplayScale();

RefPtr<JSON::Object> jsonWithLowercaseKeys(RefPtr<JSON::Object>);
RefPtr<JSON::Object> mergeJSON(RefPtr<JSON::Object>, RefPtr<JSON::Object>);

/// Returns a concatenated error string that combines the provided information into a single, descriptive message.
String toErrorString(const String& callingAPIName, const String& sourceKey, String underlyingErrorString, ...);

/// Returns an error for Expected results in CompletionHandler.
template<typename... Args>
Unexpected<WebExtensionError> toWebExtensionError(const String& callingAPIName, const String& sourceKey, const String& underlyingErrorString, Args&&... args)
{
    return makeUnexpected(toErrorString(callingAPIName, sourceKey, underlyingErrorString, std::forward<Args>(args)...));
}

#ifdef __OBJC__

/// Verifies that a dictionary:
///  - Contains a required set of string keys, as listed in `requiredKeys`, all other types specified in `keyTypes` are optional keys.
///  - Has values that are the appropriate type for each key, as specified by `keyTypes`. The keys in this dictionary
///    correspond to keys in the original dictionary being validated, and the values in `keyTypes` may be:
///     - A `Class`, that the value in the original dictionary must be a kind of.
///     - An `NSArray` containing one class, specifying that the value in the original dictionary must be an array with elements that are a kind of the specified class.
///     - An `NSOrderedSet` containing one or more classes or arrays, specifying the value in the dictionary should be of any classes in the set or their subclasses.
///  - The `callingAPIName` and `sourceKey` parameters are used to reference the object within a larger context. When an error occurs, this key helps identify the source of the problem in the `outExceptionString`.
/// If the dictionary is valid, returns `YES`. Otherwise returns `NO` and sets `outExceptionString` to a message describing what validation failed.
bool validateDictionary(NSDictionary<NSString *, id> *, NSString *sourceKey, NSArray<NSString *> *requiredKeys, NSDictionary<NSString *, id> *keyTypes, NSString **outExceptionString);

/// Verifies a single object against a certain type criteria:
///  - Checks that the object matches the type defined in `valueTypes`. The `valueTypes` can be:
///     - A `Class`, indicating the object should be of this class or its subclass.
///     - An `NSArray` containing one class, meaning the object must be an array with elements that are a kind of the specified class.
///     - An `NSOrderedSet` containing one or more classes or arrays, specifying that the object should be of any class in the set or their subclasses.
///  - The `callingAPIName` and `sourceKey` parameters are used to reference the object within a larger context. When an error occurs, this key helps identify the source of the problem in the `outExceptionString`.
/// If the object is valid, returns `YES`. Otherwise returns `NO` and sets `outExceptionString` to a message describing what validation failed.
bool validateObject(NSObject *, NSString *sourceKey, id valueTypes, NSString **outExceptionString);

/// Returns an error object that combines the provided information into a single, descriptive message.
JSObjectRef toJSError(JSContextRef, NSString *callingAPIName, NSString *sourceKey, NSString *underlyingErrorString);

/// Returns a rejected Promise object that combines the provided information into a single, descriptive error message.
JSObjectRef toJSRejectedPromise(JSContextRef, NSString *callingAPIName, NSString *sourceKey, NSString *underlyingErrorString);

NSString *toWebAPI(NSLocale *);

/// Returns the storage size of a string.
size_t storageSizeOf(NSString *);

/// Returns the storage size of all of the key value pairs in a dictionary.
size_t storageSizeOf(NSDictionary<NSString *, NSString *> *);

/// Returns true if the size of any item in the dictionary exceeds the given quota.
bool anyItemsExceedQuota(NSDictionary *, size_t quota, NSString **outKeyWithError = nullptr);

enum class UseNullValue : bool { No, Yes };

template<typename T>
id toWebAPI(const std::optional<T>& result, UseNullValue useNull = UseNullValue::Yes)
{
    if (!result)
        return useNull == UseNullValue::Yes ? NSNull.null : nil;
    return toWebAPI(result.value());
}

template<typename T>
NSArray *toWebAPI(const Vector<T>& items)
{
    return createNSArray(items, [](const T& item) {
        return toWebAPI(item);
    }).get();
}

inline NSNumber *toWebAPI(size_t index)
{
    return index != notFound ? @(index) : @(std::numeric_limits<double>::quiet_NaN());
}

#endif // __OBJC__

Markable<WTF::UUID> toDocumentIdentifier(WebFrame&);

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
