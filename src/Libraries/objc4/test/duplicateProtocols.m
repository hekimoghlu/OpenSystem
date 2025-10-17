/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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


#include "test.h"

// This test assumes
// - that on launch we don't have NSCoding, NSSecureCoding, or NSDictionary, ie, libSystem doesn't contain those
// - that after dlopening CF, we get NSDictionary and it conforms to NSSecureCoding which conforms to NSCoding
// - that our NSCoding will be used if we ask either NSSecureCoding or NSDictionary if they conform to our test protocol

@protocol NewNSCodingSuperProto
@end

@protocol NSCoding <NewNSCodingSuperProto>
@end

int main()
{
	// Before we dlopen, make sure we are using our NSCoding, not the shared cache version
	Protocol* codingSuperProto = objc_getProtocol("NewNSCodingSuperProto");
	Protocol* codingProto = objc_getProtocol("NSCoding");
	if (@protocol(NewNSCodingSuperProto) != codingSuperProto) fail("Protocol mismatch");
	if (@protocol(NSCoding) != codingProto) fail("Protocol mismatch");
	if (!protocol_conformsToProtocol(codingProto, codingSuperProto)) fail("Our NSCoding should conform to NewNSCodingSuperProto");

	// Also make sure we don't yet have an NSSecureCoding or NSDictionary
	if (objc_getProtocol("NSSecureCoding")) fail("Test assumes we don't have NSSecureCoding yet");
	if (objc_getClass("NSDictionary")) fail("Test assumes we don't have NSDictionary yet");

    void *dl = dlopen("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation", RTLD_LAZY);
    if (!dl) fail("couldn't open CoreFoundation");
    
    // We should now have NSSecureCoding and NSDictionary
    Protocol* secureCodingProto = objc_getProtocol("NSSecureCoding");
    id dictionaryClass = objc_getClass("NSDictionary");
    if (!secureCodingProto) fail("Should have got NSSecureCoding from CoreFoundation");
    if (!dictionaryClass) fail("Should have got NSDictionary from CoreFoundation");

    // Now make sure that NSDictionary and NSSecureCoding find our new protocols
    if (!protocol_conformsToProtocol(secureCodingProto, codingProto)) fail("NSSecureCoding should conform to our NSCoding");
    if (!protocol_conformsToProtocol(secureCodingProto, codingSuperProto)) fail("NSSecureCoding should conform to our NewNSCodingSuperProto");
    if (!class_conformsToProtocol(dictionaryClass, codingProto)) fail("NSDictionary should conform to our NSCoding");
    if (!class_conformsToProtocol(dictionaryClass, codingSuperProto)) fail("NSDictionary should conform to our NewNSCodingSuperProto");
}
