/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 19, 2024.
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
#import <Foundation/Foundation.h>
#import "runtime_root.h"

@class WebUndefined;

@protocol WebScriptObject
+ (NSString *)webScriptNameForSelector:(SEL)aSelector;
+ (BOOL)isSelectorExcludedFromWebScript:(SEL)aSelector;
+ (NSString *)webScriptNameForKey:(const char *)name;
+ (BOOL)isKeyExcludedFromWebScript:(const char *)name;

+ (id)_convertValueToObjcValue:(JSC::JSValue)value originRootObject:(JSC::Bindings::RootObject*)originRootObject rootObject:(JSC::Bindings::RootObject*)rootObject;
- (id)_initWithJSObject:(JSC::JSObject*)imp originRootObject:(RefPtr<JSC::Bindings::RootObject>&&)originRootObject rootObject:(RefPtr<JSC::Bindings::RootObject>&&)rootObject;
- (JSC::JSObject *)_imp;
@end

@protocol WebUndefined
+ (WebUndefined *)undefined;
@end
