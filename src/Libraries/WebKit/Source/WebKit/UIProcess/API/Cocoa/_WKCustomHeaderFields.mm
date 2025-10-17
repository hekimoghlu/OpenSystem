/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 2, 2023.
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
#import "_WKCustomHeaderFields.h"

#import "_WKCustomHeaderFieldsInternal.h"
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/BlockPtr.h>
#import <wtf/cocoa/VectorCocoa.h>

@implementation _WKCustomHeaderFields

- (instancetype)init
{
    if (!(self = [super init]))
        return nil;
    
    API::Object::constructInWrapper<API::CustomHeaderFields>(self);
    return self;
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKCustomHeaderFields.class, self))
        return;
    _fields->API::CustomHeaderFields::~CustomHeaderFields();
    [super dealloc];
}

- (NSDictionary<NSString *, NSString *> *)fields
{
    auto& vector = _fields->fields();
    NSMutableDictionary<NSString *, NSString *> *dictionary = [NSMutableDictionary dictionaryWithCapacity:vector.size()];
    for (auto& field : vector)
        [dictionary setObject:field.value() forKey:field.name()];
    return dictionary;
}

- (void)setFields:(NSDictionary<NSString *, NSString *> *)fields
{
    Vector<WebCore::HTTPHeaderField> vector;
    vector.reserveInitialCapacity(fields.count);
    [fields enumerateKeysAndObjectsUsingBlock:makeBlockPtr([&](id key, id value, BOOL* stop) {
        if (auto field = WebCore::HTTPHeaderField::create((NSString *)key, (NSString *)value); field && startsWithLettersIgnoringASCIICase(field->name(), "x-"_s))
            vector.append(WTFMove(*field));
    }).get()];
    _fields->setFields(WTFMove(vector));
}

- (NSArray<NSString *> *)thirdPartyDomains
{
    return createNSArray(_fields->thirdPartyDomains()).autorelease();
}

- (void)setThirdPartyDomains:(NSArray<NSString *> *)thirdPartyDomains
{
    _fields->setThirdPartyDomains(makeVector<String>(thirdPartyDomains));
}

- (API::Object&)_apiObject
{
    return *_fields;
}

@end
