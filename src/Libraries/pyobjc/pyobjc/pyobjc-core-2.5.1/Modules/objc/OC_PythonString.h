/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 11, 2024.
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
#include "pyobjc.h"

/*!
 * @class       OC_PythonString
 * @abstract    Objective-C proxy class for Python str
 * @discussion  Instances of this class are used as proxies for Python 
 *              str when these are passed to Objective-C code.
 */
@interface OC_PythonString : NSString
{
	PyObject* value;
	id realObject;
}

/*!
 * @method newWithPythonObject:
 * @abstract Create a new OC_PythonString for a specific Python str
 * @param value A python str
 * @result Returns an autoreleased instance representing value
 *
 * Caller must own the GIL.
 */
+ (instancetype)stringWithPythonObject:(PyObject*)value;

/*!
 * @method initWithPythonObject:
 * @abstract Initialise a OC_PythonString for a specific Python str
 * @param value A python str
 * @result Returns self
 *
 * Caller must own the GIL.
 */
- (id)initWithPythonObject:(PyObject*)value;

/*!
 * @method dealloc
 * @abstract Deallocate the object
 */
-(void)dealloc;

/*!
 * @abstract Access the wrapped Python str
 * @result Returns a new reference to the wrapped Python str.
 */
-(PyObject*)__pyobjc_PythonObject__;

/*!
 * @abstract Access the NSString* representing the str
 * @result Returns a backing NSString* object
 */
-(id)__realObject__;

/*
 * Primitive NSString methods
 *
 */
-(NSUInteger)length;
-(unichar)characterAtIndex:(NSUInteger)index;
-(void)getCharacters:(unichar *)buffer range:(NSRange)aRange;

@end
