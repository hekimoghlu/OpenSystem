/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 6, 2021.
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
#import "KCDStructTypeDescription.h"

#ifndef KCDATA_TYPE_MAX_WITH_DESC
#define KCDATA_TYPE_MAX_WITH_DESC 0x6
#endif

@interface
KCDStructTypeDescription () {
	unsigned int _typeID;
	NSString * _name;
	NSMutableArray * _fields;
	BOOL _needDescriptionAsKey;
	BOOL _flagsRequestedMerge;
}

@end

@implementation KCDStructTypeDescription

- (id)initWithType:(unsigned int)typeID withName:(NSString *)name
{
	if ((self = [super init])) {
		_typeID               = typeID;
		_name = name;
		_needDescriptionAsKey = NO;
		if (typeID >= 0x1 && typeID <= KCDATA_TYPE_MAX_WITH_DESC)
			_needDescriptionAsKey = YES;

		_fields              = [[NSMutableArray alloc] init];
		_flagsRequestedMerge = NO;
		return self;
	}
	return NULL;
}

- (void)addFieldBasicType:(KCDBasicTypeDescription *)fieldType
{
	[_fields addObject:fieldType];
}

- (void)setFlagsRequestedMerge
{
	_flagsRequestedMerge = YES;
}

- (NSDictionary *)parseData:(void *)dataBuffer ofLength:(uint32_t)length
{
	NSMutableDictionary * retval = [[NSMutableDictionary alloc] init];
	for (KCDataType * fi in _fields) {
		NSDictionary * _d = [fi parseData:dataBuffer ofLength:length];
		if (!_d) {
			return nil;
		}
		for (NSString * k in [_d keyEnumerator]) {
			retval[k] = _d[k];
		}
	}
    if (_typeID == KCDATA_TYPE_TYPEDEFINTION){
        uint32_t elem_size = sizeof(struct kcdata_subtype_descriptor);
        uint32_t elem_count = (length - offsetof(struct kcdata_type_definition, kct_elements))/elem_size;
        NSMutableArray * fields_array = [NSMutableArray arrayWithCapacity:elem_count];
        struct kcdata_subtype_descriptor *fields_dsc = (struct kcdata_subtype_descriptor *) ((uintptr_t)dataBuffer + offsetof(struct kcdata_type_definition, kct_elements));
        int i = 0;
        for (i = 0; i < elem_count; i++) {
            KCDBasicTypeDescription * tmpdsc = [[KCDBasicTypeDescription alloc] initWithKCTypeDesc:&fields_dsc[i]];
            NSString *field_desc_str = [tmpdsc description];
            
            [fields_array addObject:field_desc_str];
        }
        retval[@"fields"] = fields_array;
    }
	if (_needDescriptionAsKey) {
		NSString * desc = retval[@"desc"];
		NSObject * obj = retval[@"data"];
		retval[desc] = obj;
		[retval removeObjectForKey:@"desc"];
		[retval removeObjectForKey:@"data"];
	}
	return retval;
}

- (BOOL)shouldMergeData
{
	/*
	 * If this is a type where the kcdata item itself carries the key name, or
	 * KCS_SUBTYPE_FLAGS_MERGE was used to define the type, then a member of
	 * this type should have it's dict merged into the parent container,
	 * instead of being represented as typename => dict.
	 */
	return _needDescriptionAsKey || _flagsRequestedMerge;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"type: %d => \"%@\" ", _typeID, _name];
}

- (NSString *)name
{
	return _name;
}

- (uint32_t)count
{
	return (uint32_t)[_fields count];
}

- (unsigned int)typeID
{
	return _typeID;
}

@end
