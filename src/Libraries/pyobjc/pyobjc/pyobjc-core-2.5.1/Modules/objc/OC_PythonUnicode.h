/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 30, 2023.
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
 * @class       OC_PythonUnicode
 * @abstract    Objective-C proxy class for Python unicode
 * @discussion  Instances of this class are used as proxies for Python 
 *              unicode when these are passed to Objective-C code.
 */
@interface OC_PythonUnicode : NSString
{
	PyObject* value;
	id realObject;

#ifdef PyObjC_STR_CACHE_IMP
	/* Cache IMPs for proxied methods, for slightly better efficiency */
	NSUInteger (*imp_length)(id, SEL);
	unichar (*imp_charAtIndex)(id, SEL, NSUInteger);
	void (*imp_getCharacters)(id, SEL, unichar*, NSRange);
#endif /* PyObjC_STR_CACHE_IMP */
}

/*!
 * @method newWithPythonObject:
 * @abstract Create a new OC_PythonUnicode for a specific Python unicode
 * @param value A python unicode
 * @result Returns an autoreleased instance representing value
 *
 * Caller must own the GIL.
 */
+(id)unicodeWithPythonObject:(PyObject*)value;

/*!
 * @method initWithPythonObject:
 * @abstract Initialise a OC_PythonUnicode for a specific Python unicode
 * @param value A python unicode
 * @result Returns self
 *
 * Caller must own the GIL.
 */
-(id)initWithPythonObject:(PyObject*)value;

/*!
 * @method dealloc
 * @abstract Deallocate the object
 */
-(void)dealloc;

/*!
 * @abstract Access the wrapped Python unicode
 * @result Returns a new reference to the wrapped Python unicode.
 */
-(PyObject*)__pyobjc_PythonObject__;

/*
 * Primitive NSString methods
 *
 */
-(NSUInteger)length;
-(unichar)characterAtIndex:(NSUInteger)index;
-(void)getCharacters:(unichar *)buffer range:(NSRange)aRange;

@end
