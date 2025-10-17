/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 20, 2024.
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
#import "OC_NSBundleHack.h"
#import "objc_util.h"

static id (*bundleForClassIMP)(id, SEL, Class);

@implementation OC_NSBundleHackCheck
+(NSBundle*)bundleForClass
{
	return [NSBundle bundleForClass:[NSObject class]];
}
@end

@implementation OC_NSBundleHack
+(NSBundle*)bundleForClass:(Class)aClass
{
	static NSBundle* mainBundle = nil;
	static NSMapTable* bundleCache = nil;
	if (unlikely(!mainBundle)) {
		mainBundle = [[NSBundle mainBundle] retain];
	}
	if (unlikely(!bundleCache)) {
		bundleCache = NSCreateMapTable(
			PyObjCUtil_PointerKeyCallBacks,
			PyObjCUtil_ObjCValueCallBacks,
			PYOBJC_EXPECTED_CLASS_COUNT);
	}
	if (!aClass) {
		return mainBundle;
	}
	id rval = (id)NSMapGet(bundleCache, (const void *)aClass);
	if (rval) {
		return rval;
	}
	rval = bundleForClassIMP(self, @selector(bundleForClass:), aClass);
	if (rval == mainBundle) {
		Class base_isa = aClass;
		Class nsobject_isa = object_getClass([NSObject class]);
		while (base_isa != nsobject_isa) {
			Class next_isa = object_getClass(base_isa);
			if (!next_isa || next_isa == base_isa) {
				break;
			}
			base_isa = next_isa;
		}
		if (base_isa == nsobject_isa) {
			if ([(id)aClass respondsToSelector:@selector(bundleForClass)]) {
				rval = [(id)aClass performSelector:@selector(bundleForClass)];
			}
		}
	}
	NSMapInsert(bundleCache, (const void *)aClass, (const void *)rval);
	return rval;
}

+(void)installBundleHack
{
	if ([[NSBundle bundleForClass:[NSObject class]] isEqual:[NSBundle bundleForClass:[OC_NSBundleHackCheck class]]]) {
		// implementation is already fine
		return;
	}
	bundleForClassIMP = (id (*)(id, SEL, Class))[NSBundle methodForSelector:@selector(bundleForClass:)];

	Method method = class_getInstanceMethod(
			object_getClass([NSBundle class]),
			@selector(bundleForClass:));
	if (method == NULL) {
		class_addMethod(
			object_getClass([NSBundle class]),
			@selector(bundleForClass:),
			[self methodForSelector:@selector(bundleForClass:)],
		        "@@:#");
	} else {
		method_setImplementation(method, 
			[self methodForSelector:@selector(bundleForClass:)]
		);
	}
}
@end
