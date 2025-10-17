/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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
#import "ArgumentCodersCocoa.h"

@implementation WKKeyedCoder {
    RetainPtr<NSMutableDictionary> m_dictionary;
    bool m_failedDecoding;
}

- (instancetype)init
{
    if (!(self = [super init]))
        return nil;

    m_dictionary = adoptNS([NSMutableDictionary new]);
    return self;
}

- (instancetype)initWithDictionary:(NSDictionary *)dictionary
{
    if (!(self = [super init]))
        return nil;

    m_dictionary = adoptNS([dictionary mutableCopy]);
    return self;
}

- (BOOL)allowsKeyedCoding
{
    return YES;
}

- (BOOL)requiresSecureCoding
{
    return YES;
}

- (void)encodeObject:(id)object forKey:(NSString *)key
{
    if (!object)
        return;

    if (!IPC::isSerializableValue(object)) {
        ASSERT_NOT_REACHED_WITH_MESSAGE("WKKeyedCoder attempt to encode object of unsupported type %s", object_getClassName(object));
        return;
    }

    [m_dictionary setObject:object forKey:key];
}

- (BOOL)containsValueForKey:(NSString *)key
{
    return !![m_dictionary objectForKey:key];
}

- (id)decodeObjectOfClass:(Class)aClass forKey:(NSString *)key
{
    if (m_failedDecoding)
        return nil;

    id object = [m_dictionary objectForKey:key];
    if (object && ![object isKindOfClass:aClass]) {
        m_failedDecoding = YES;
        return nil;
    }

    return object;
}

- (id)decodeObjectOfClasses:(NSSet<Class> *)classes forKey:(NSString *)key
{
    if (m_failedDecoding)
        return nil;

    id object = [m_dictionary objectForKey:key];
    if (!object)
        return nil;
    for (id aClass in classes) {
        if ([object isKindOfClass:aClass])
            return object;
    }

    m_failedDecoding = YES;
    return nil;
}

- (id)decodeObjectForKey:(NSString *)key
{
    return [m_dictionary objectForKey:key];
}

- (void)encodeBool:(BOOL)value forKey:(NSString *)key
{
    [self encodeObject:@(value) forKey:key];
}

- (BOOL)decodeBoolForKey:(NSString *)key
{
    return [[self decodeObjectOfClass:[NSNumber class] forKey:key] boolValue];
}

- (void)encodeInt64:(int64_t)value forKey:(NSString *)key
{
    [self encodeObject:@(value) forKey:key];
}

- (int64_t)decodeInt64ForKey:(NSString *)key
{
    return [[self decodeObjectOfClass:[NSNumber class] forKey:key] longLongValue];
}

- (void)encodeInteger:(NSInteger)value forKey:(NSString *)key
{
    [self encodeObject:@(value) forKey:key];
}

- (NSInteger)decodeIntegerForKey:(NSString *)key
{
    return [[self decodeObjectOfClass:[NSNumber class] forKey:key] integerValue];
}

- (NSDictionary *)accumulatedDictionary
{
    return m_dictionary.get();
}

@end // @implementation WKKeyedCoder
