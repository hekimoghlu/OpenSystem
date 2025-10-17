/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 18, 2023.
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
#include "SecTransform.h"
#include "SecCustomTransform.h"
#include "SecExternalSourceTransform.h"
#include <dispatch/dispatch.h>

CFStringRef external_source_name = CFSTR("com.apple.security.external_source");

static SecTransformInstanceBlock SecExternalSourceTransformCreateBlock(CFStringRef name, SecTransformRef newTransform, SecTransformImplementationRef ref)
{
	return Block_copy(^{ 
		SecTransformCustomSetAttribute(ref, kSecTransformInputAttributeName, kSecTransformMetaAttributeRequired, kCFBooleanFalse);
		
		SecTransformAttributeRef out = SecTranformCustomGetAttribute(ref, kSecTransformOutputAttributeName, kSecTransformMetaAttributeRef);
		
		SecTransformSetAttributeAction(ref, kSecTransformActionAttributeNotification, kSecTransformInputAttributeName, ^(SecTransformAttributeRef attribute, CFTypeRef value) {
			SecTransformCustomSetAttribute(ref, out, kSecTransformMetaAttributeValue, value);
			return (CFTypeRef)NULL;
		});
		
		return (CFErrorRef)NULL;
	});
}

SecTransformRef SecExternalSourceTransformCreate(CFErrorRef* error)
{
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		SecTransformRegister(external_source_name, SecExternalSourceTransformCreateBlock, error);
	});
	
	return SecTransformCreate(external_source_name, error);
}
