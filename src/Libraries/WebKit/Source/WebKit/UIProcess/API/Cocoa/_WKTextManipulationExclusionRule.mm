/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 29, 2023.
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
#import "_WKTextManipulationExclusionRule.h"

#import <wtf/RetainPtr.h>

@implementation _WKTextManipulationExclusionRule {
    BOOL _isExclusion;
    RetainPtr<NSString> _elementName;
    RetainPtr<NSString> _attributeName;
    RetainPtr<NSString> _attributeValue;
    RetainPtr<NSString> _className;
}

- (instancetype)initExclusion:(BOOL)exclusion forElement:(NSString *)localName
{
    if (!(self = [super init]))
        return nil;

    _isExclusion = exclusion;
    _elementName = localName;
    
    return self;
}

- (instancetype)initExclusion:(BOOL)exclusion forAttribute:(NSString *)name value:(NSString *)value
{
    if (!(self = [super init]))
        return nil;

    _isExclusion = exclusion;
    _attributeName = name;
    _attributeValue = value;

    return self;
}

- (instancetype)initExclusion:(BOOL)exclusion forClass:(NSString *)className
{
    if (!(self = [super init]))
        return nil;

    _isExclusion = exclusion;
    _className = className;

    return self;
}

- (NSString *)elementName
{
    return _elementName.get();
}

- (NSString *)attributeName
{
    return _attributeName.get();
}

- (NSString *)attributeValue
{
    return _attributeValue.get();
}

- (NSString *)className
{
    return _className.get();
}

@end

