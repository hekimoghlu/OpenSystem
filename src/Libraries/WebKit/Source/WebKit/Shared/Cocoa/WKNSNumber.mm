/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 23, 2022.
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
#import "WKNSNumber.h"

using namespace WebKit;

@implementation WKNSNumber {
    union {
        API::ObjectStorage<API::Boolean> _boolean;
        API::ObjectStorage<API::Double> _double;
        API::ObjectStorage<API::UInt64> _uint64;
        API::ObjectStorage<API::Int64> _int64;
    } _number;
}

- (void)dealloc
{
    switch (_type) {
    case API::Object::Type::Boolean:
        _number._boolean->~Boolean();
        break;

    case API::Object::Type::Double:
        _number._double->~Double();
        break;

    case API::Object::Type::UInt64:
        _number._uint64->~UInt64();
        break;

    case API::Object::Type::Int64:
        _number._int64->~Int64();
        break;

    default:
        ASSERT_NOT_REACHED();
    }

    [super dealloc];
}

// MARK: NSValue primitive methods

- (const char *)objCType
{
    switch (_type) {
    case API::Object::Type::Boolean:
        return @encode(bool);
        break;

    case API::Object::Type::Double:
        return @encode(double);
        break;

    case API::Object::Type::UInt64:
        return @encode(uint64_t);
        break;

    case API::Object::Type::Int64:
        return @encode(int64_t);
        break;

    default:
        ASSERT_NOT_REACHED();
    }

    return nullptr;
}

- (void)getValue:(void *)value
{
    switch (_type) {
    case API::Object::Type::Boolean:
        *reinterpret_cast<bool*>(value) = _number._boolean->value();
        break;

    case API::Object::Type::Double:
        *reinterpret_cast<double*>(value) = _number._double->value();
        break;

    case API::Object::Type::UInt64:
        *reinterpret_cast<uint64_t*>(value) = _number._uint64->value();
        break;

    case API::Object::Type::Int64:
        *reinterpret_cast<int64_t*>(value) = _number._int64->value();
        break;

    default:
        ASSERT_NOT_REACHED();
    }
}

// MARK: NSNumber primitive methods

- (char)charValue
{
    if (_type == API::Object::Type::Boolean)
        return _number._boolean->value();

    return super.charValue;
}

- (double)doubleValue
{
    if (_type == API::Object::Type::Double)
        return _number._double->value();

    return super.doubleValue;
}

- (unsigned long long)unsignedLongLongValue
{
    if (_type == API::Object::Type::UInt64)
        return _number._uint64->value();

    return super.unsignedLongLongValue;
}

- (long long)longLongValue
{
    if (_type == API::Object::Type::Int64)
        return _number._int64->value();

    return super.longLongValue;
}

// MARK: NSCopying protocol implementation

- (id)copyWithZone:(NSZone *)zone
{
    return [self retain];
}

// MARK: WKObject protocol implementation

- (API::Object&)_apiObject
{
    switch (_type) {
    case API::Object::Type::Boolean:
        return *_number._boolean;
        break;

    case API::Object::Type::Double:
        return *_number._double;
        break;

    case API::Object::Type::UInt64:
        return *_number._uint64;
        break;

    case API::Object::Type::Int64:
        return *_number._int64;
        break;

    default:
        ASSERT_NOT_REACHED();
    }

    return *_number._boolean;
}

@end
