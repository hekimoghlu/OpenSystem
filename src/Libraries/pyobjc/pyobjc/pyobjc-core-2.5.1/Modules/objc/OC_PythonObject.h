/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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
#ifndef _OC_PythonObject_H
#define _OC_PythonObject_H

#import <Foundation/NSProxy.h>
#import <Foundation/NSDictionary.h>
#import <Foundation/NSMethodSignature.h>

extern PyObject* PyObjC_Encoder;
extern PyObject* PyObjC_Decoder;
extern PyObject* PyObjC_CopyFunc;

extern void PyObjC_encodeWithCoder(PyObject* pyObject, NSCoder* coder);




@interface OC_PythonObject : NSProxy  <NSCopying>
{
  PyObject *pyObject;
}

+ (int)wrapPyObject:(PyObject *)argument toId:(id *)datum;
+ (id <NSObject>)objectWithPythonObject:(PyObject *) obj;
+ (id)depythonifyTable;
+ (id)pythonifyStructTable;
+ (PyObject *)__pythonifyStruct:(PyObject *) obj withType:(const char *) type length:(Py_ssize_t) length;
+ (id <NSObject>)objectWithCoercedPyObject:(PyObject *) obj;
- (id)initWithPyObject:(PyObject *) obj;

/*!
 * @method pyObject
 * @result Returns a borrowed reference to the wrapped object
 */
- (PyObject*) pyObject;

/*!
 * @method __pyobjc_PythonObject__
 * @result Returns a new reference to the wrapped object
 * @discussion
 * 	This method is part of the implementation of objc_support.m,
 * 	see that file for details.
 */
- (PyObject*) __pyobjc_PythonObject__;
- (void) forwardInvocation:(NSInvocation *) invocation;
- (BOOL) respondsToSelector:(SEL) aSelector;
- (NSMethodSignature *) methodSignatureForSelector:(SEL) selector;
- (void) doesNotRecognizeSelector:(SEL) aSelector;

/* NSObject protocol */
- (NSUInteger)hash;
- (BOOL)isEqual:(id)anObject;
/* NSObject methods */
- (NSComparisonResult)compare:(id)other;

/* Key-Value Coding support */
+ (BOOL)useStoredAccessor;
+ (BOOL)accessInstanceVariablesDirectly;
- (id)valueForKey:(NSString*) key;
- (NSDictionary*) valuesForKeys: (NSArray*)keys;
- (id)valueForKeyPath: (NSString*) keyPath;
- (id)storedValueForKey: (NSString*) key;
- (void)takeValue: value forKey: (NSString*) key;
- (void)setValue: value forKey: (NSString*) key;
- (void)setValue: value forKeyPath: (NSString*) key;
- (void)takeStoredValue: value forKey: (NSString*) key;
- (void)takeValue: value forKeyPath: (NSString*) keyPath;
- (void)takeValuesFromDictionary: (NSDictionary*) aDictionary;
- (void)setValuesForKeysWithDictionary: (NSDictionary*) aDictionary;
- (void)unableToSetNilForKey: (NSString*) key;
- (void)valueForUndefinedKey: (NSString*) key;
- (void)handleTakeValue: value forUnboundKey: (NSString*) key;
- (void)setValue: value forUndefinedKey: (NSString*) key;

/* These two are only present to *disable* coding, not implement it */
- (void)encodeWithCoder:(NSCoder*)coder;
- (id)initWithCoder:(NSCoder*)coder;
+ (id)classFallbacksForKeyedArchiver;
-(NSObject*)replacementObjectForArchiver:(NSArchiver*)archiver;
-(NSObject*)replacementObjectForKeyedArchiver:(NSKeyedArchiver*)archiver;
-(NSObject*)replacementObjectForCoder:(NSCoder*)archiver;
-(NSObject*)replacementObjectForPortCoder:(NSPortCoder*)archiver;
-(Class)classForArchiver;
-(Class)classForKeyedArchiver;
+(Class)classForUnarchiver;
+(Class)classForKeyedUnarchiver;
-(Class)classForCoder;
-(Class)classForPortCoder;
-(id)awakeAfterUsingCoder:(NSCoder*)coder;

@end /* OC_PythonObject class interface */

#endif /* _OC_PythonObject_H */
