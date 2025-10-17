/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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
#import "pyobjc.h"
#import <Foundation/Foundation.h>

/*!
 * @class       OC_PythonData
 * @abstract    Objective-C proxy class for Python buffers
 * @discussion  Instances of this class are used as proxies for Python 
 *          buffers when these are passed to Objective-C code. Because 
 *          this class is a subclass of NSData, Python buffers
 *          (except str, unicode) can be used everywhere where NSData
 *          is expected.
 */
@interface OC_PythonData : NSData
{
	PyObject* value;

	/* XXX: why are these here? These fields don't seem to be necessary
	 * at all!
	 */
	Py_ssize_t buffer_len;
	const void *buffer;
}

/*!
 * @method newWithPythonObject:
 * @abstract Create a new OC_PythonData for a specific Python buffer
 * @param value A python buffer
 * @result Returns an autoreleased instance representing value
 *
 * Caller must own the GIL.
 */
+(OC_PythonData*)dataWithPythonObject:(PyObject*)value;

/*!
 * @method initWithPythonObject:
 * @abstract Initialise a OC_PythonData for a specific Python buffer
 * @param value A python buffer
 * @result Returns self
 *
 * Caller must own the GIL.
 */
-(OC_PythonData*)initWithPythonObject:(PyObject*)value;

/*!
 * @method dealloc
 * @abstract Deallocate the object
 */
-(void)dealloc;

/*!
 * @method dealloc
 * @abstract Access the wrapped Python buffer
 * @result Returns a new reference to the wrapped Python buffer.
 */
-(PyObject*)__pyobjc_PythonObject__;

/*!
 * @method length
 * @result Returns the length of the wrapped Python buffer
 */
-(NSUInteger)length;

/*!
 * @method bytes
 * @result Returns a pointer to the contents of the Python buffer
 */
-(const void *)bytes;

@end
