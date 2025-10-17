/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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
#import "WKWebExtensionDataRecordInternal.h"

#import "WKWebExtensionDataTypeInternal.h"

NSErrorDomain const WKWebExtensionDataRecordErrorDomain = @"WKWebExtensionDataRecordErrorDomain";

@implementation WKWebExtensionDataRecord

#if ENABLE(WK_WEB_EXTENSIONS)

WK_OBJECT_DEALLOC_IMPL_ON_MAIN_THREAD(WKWebExtensionDataRecord, WebExtensionDataRecord, _webExtensionDataRecord);

- (BOOL)isEqual:(id)object
{
    if (self == object)
        return YES;

    auto *other = dynamic_objc_cast<WKWebExtensionDataRecord>(object);
    if (!other)
        return NO;

    return *_webExtensionDataRecord == *other->_webExtensionDataRecord;
}

- (NSString *)displayName
{
    return _webExtensionDataRecord->displayName();
}

- (NSString *)uniqueIdentifier
{
    return _webExtensionDataRecord->uniqueIdentifier();
}

- (NSSet<WKWebExtensionDataType> *)containedDataTypes
{
    return toAPI(self._protectedWebExtensionDataRecord->types());
}

- (NSUInteger)totalSizeInBytes
{
    return self._protectedWebExtensionDataRecord->totalSize();
}

- (NSUInteger)sizeInBytesOfTypes:(NSSet<WKWebExtensionDataType> *)dataTypes
{
    return self._protectedWebExtensionDataRecord->sizeOfTypes(WebKit::toWebExtensionDataTypes(dataTypes));
}

- (NSArray<NSError *> *)errors
{
    return self._protectedWebExtensionDataRecord->errors();
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_webExtensionDataRecord;
}

- (WebKit::WebExtensionDataRecord&)_webExtensionDataRecord
{
    return *_webExtensionDataRecord;
}

- (Ref<WebKit::WebExtensionDataRecord>)_protectedWebExtensionDataRecord
{
    return *_webExtensionDataRecord;
}

#else // ENABLE(WK_WEB_EXTENSIONS)

- (NSString *)displayName
{
    return nil;
}

- (NSString *)uniqueIdentifier
{
    return nil;
}

- (NSSet<WKWebExtensionDataType> *)containedDataTypes
{
    return nil;
}

- (NSUInteger)totalSizeInBytes
{
    return 0;
}

- (NSUInteger)sizeInBytesOfTypes:(NSSet<WKWebExtensionDataType> *)dataTypes
{
    return 0;
}

- (NSArray<NSError *> *)errors
{
    return nil;
}

#endif // ENABLE(WK_WEB_EXTENSIONS)

@end

#if ENABLE(WK_WEB_EXTENSIONS)

namespace WebKit {

static std::optional<Ref<WebExtensionDataRecord>> makeVectorElement(const Ref<WebExtensionDataRecord>*, id arrayElement)
{
    if (auto *record = dynamic_objc_cast<WKWebExtensionDataRecord>(arrayElement))
        return Ref { record._webExtensionDataRecord };
    return std::nullopt;
}

static RetainPtr<id> makeNSArrayElement(const Ref<WebExtensionDataRecord>& vectorElement)
{
    return vectorElement->wrapper();
}

Vector<Ref<WebExtensionDataRecord>> toWebExtensionDataRecords(NSArray *records)
{
    return makeVector<Ref<WebExtensionDataRecord>>(records);
}

NSArray *toAPI(const Vector<Ref<WebExtensionDataRecord>>& records)
{
    return createNSArray(records).get();
}

NSError *createDataRecordError(WKWebExtensionDataRecordError error, NSString *debugDescription)
{
    NSDictionary *userInfo = debugDescription ? @{ NSDebugDescriptionErrorKey: debugDescription } : @{ };
    return [NSError errorWithDomain:WKWebExtensionDataRecordErrorDomain code:error userInfo:userInfo];
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
