/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 28, 2022.
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

#ifndef OBJC_OBJC_H_
#define OBJC_OBJC_H_

#define OBJC_ARC_UNAVAILABLE __attribute__((unavailable("not available in automatic reference counting mode")))
#define NS_AUTOMATED_REFCOUNT_UNAVAILABLE OBJC_ARC_UNAVAILABLE

#ifdef __LP64__
typedef unsigned long NSUInteger;
typedef long NSInteger;
#else
typedef unsigned int NSUInteger;
typedef int NSInteger;
#endif

typedef __typeof__(__objc_yes) BOOL;

typedef struct objc_selector    *SEL;
SEL sel_registerName(const char *str);
BOOL sel_isEqual(SEL sel1, SEL sel2);

void NSDeallocateObject(id object) NS_AUTOMATED_REFCOUNT_UNAVAILABLE;

#undef NS_AUTOMATED_REFCOUNT_UNAVAILABLE

#define OBJC_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define OBJC_OPTIONS(_type, _name) enum _name : _type _name; enum _name : _type

typedef OBJC_ENUM(int, objc_abi) {
  objc_v1 = 0,
  objc_v2 = 2
};

typedef OBJC_OPTIONS(int, objc_flags) {
  objc_taggedPointer = 1 << 0,
  objc_languageRefcount = 1 << 1
};

#endif
