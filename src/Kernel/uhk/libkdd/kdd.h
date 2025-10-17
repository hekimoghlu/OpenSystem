/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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
#ifndef _KDD_H_
#define _KDD_H_

#import <Foundation/Foundation.h>
#import <kcdata.h>

/*!
 * @class KCDataType
 * A basic abstraction that allows for parsing data provided by kernel chunked
 * data library.
 *
 * @discussion
 * Each type object has a name and a method to parse and populate data in memory to
 * a dictionary. The dictionary will have keys as NSStrings and values could be NSObject
 *
 */
@interface KCDataType : NSObject
- (NSDictionary * _Nullable)parseData:(void * _Nonnull)dataBuffer ofLength:(uint32_t)length NS_RETURNS_RETAINED;
- (NSString * _Nonnull)name;
- (unsigned int)typeID;
- (BOOL) shouldMergeData;
@end

/*!
 * @function getKCDataTypeForID
 *
 * @abstract
 * Find a type description for give TypeID
 *
 * @param typeID
 * A unsinged int type specified by the KCDATA.
 *
 * @discussion
 * This routine queries the system for a give type. If a known type description is found it will be used to
 * initialize a KCDataType object. If no known type is found it assumes the data is uint8_t[].
 */
KCDataType * _Nullable getKCDataTypeForID(uint32_t typeID);

/*!
 * @function KCDataTypeNameForID
 *
 * @abstract
 * Get a name for the type.
 *
 * @param typeID
 * A unsinged int type specified by the KCDATA.
 *
 * @return NSString *
 * Returns name of the type. If a type is not found the return
 * value will be string object of the passed value.
 */
NSString * _Nonnull KCDataTypeNameForID(uint32_t typeID) NS_RETURNS_NOT_RETAINED;

/*!
 * @function parseKCDataArray
 *
 * @abstract
 * Parse the given KCDATA buffer as an Array of element. The buffer should begin with header
 * of type KCDATA_TYPE_ARRAY.
 *
 * @param iter
 * An iterator into the input buffer
 *
 * @param error
 * Error return.
 *
 * @return
 * A dictionary with  key specifying name of the type of each elements and value is an Array of data.
 *
 */

NSMutableDictionary * _Nullable parseKCDataArray(kcdata_iter_t iter, NSError * _Nullable * _Nullable error) NS_RETURNS_RETAINED;

/*!
 * @function parseKCDataContainer
 *
 * @abstract
 * Parse the given KCDATA buffer as a container and convert each sub structures as fields in a dictionary.
 *
 * @param iter
 * A pointer to an iterator into the input buffer.  The iterator will be updated
 * to point at the container end marker.
 *
 * @param error
 * Error return.
 *
 * @return NSDictionary *
 * containing each field and potentially sub containers within the provided container.
 *
 * @discussion
 * This function tries to parse one container. If it encounters sub containers
 * they will be parsed and collected within the same dictionary.
 * Other data type fields will also be parsed based on their type.
 *
 */

NSMutableDictionary * _Nullable parseKCDataContainer(kcdata_iter_t * _Nonnull iter, NSError * _Nullable * _Nullable error) NS_RETURNS_RETAINED;

/*!
 * @function parseKCDataBuffer
 *
 * @abstract
 * Parse complete KCDATA buffer into NSMutableDictionary. Depending on the size of buffer and elements
 * this routine makes allocations for objects and strings.
 *
 * @param dataBuffer
 * A pointer in memory where KCDATA is allocated. The data should be of type
 * kcdata_item_t and have KCDATA_BUFFER_BEGIN_* tags (see kern_cdata.h)
 *
 * @param size
 * Size of the buffer as provided by kernel api.
 *
 * @return NSDictionary *
 * Dictionary with key:value pairs for each data item. KCDATA_TYPE_ARRAY and KCDATA_TYPE_CONTAINERS will
 * grouped and recursed as much possible. For unknown types NSData object is returned with "Type_0x123"
 * as keys.
 *
 * @discussion
 * This function tries to parse KCDATA buffer with known type description. If an error occurs,
 * NULL is returned, and error (if not NULL) will have the error string.
 *
 * Iff the buffer does begin with a known kcdata magic number, the error code
 * will be KERN_INVALID_VALUE.
 *
 */
NSDictionary * _Nullable parseKCDataBuffer(void * _Nonnull dataBuffer, uint32_t size, NSError * _Nullable * _Nullable error) NS_RETURNS_RETAINED;


#endif /* _KDD_H_ */
