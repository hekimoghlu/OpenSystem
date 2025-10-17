/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 23, 2024.
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
//
// SecCFTypes.h - CF runtime interface
//
#ifndef _SECURITY_SECCFTYPES_H_
#define _SECURITY_SECCFTYPES_H_

#include <CoreFoundation/CFRuntime.h>
#include <security_utilities/globalizer.h>
#include <security_utilities/cfclass.h>

namespace Security
{

namespace KeychainCore
{

/* Singleton that registers all the CFClass instances with the CFRuntime.

   To make something a CFTypeRef you need to make the actual object inheirit from SecCFObject and provide implementation of the virtual functions in that class.
   
   In addition to that you need to define an opque type for the C API like:
   typedef struct __OpaqueYourObject *YourObjectRef;

   Add an instance of CFClass to the public section of SecCFTypes below to get it registered with the CFRuntime.
   CFClass yourObject;

   XXX
   In your C++ code you should use SecPointer<YourObject> to refer to instances of your class.  SecPointers are just like autopointers and implement * and -> semantics.  They refcount the underlying object.  So to create an instance or your new object you would do something like:
   
       SecPointer<YourObject> instance(new YourObject());

   SecPointers have copy semantics and if you subclass SecPointer and define a operator < on the subclass you can even safely store instances of your class in stl containers.

	Use then like this:
		instance->somemethod();
	or if you want a reference to the underlying object:
		YourObject &object = *instance;
	if you want a pointer to the underlying object:
		YourObject *object = instance.get();

	In the API glue you will need to use:
		SecPointer<YourObject> instance;
		[...] get the instance somehow
		return instance->handle();
		to return an opaque handle (the is a CFTypeRef) to your object.
		
	when you obtain an object as input use:
		SecYourObjectRef ref;
		SecPointer<YourObject> instance = YourObject::required(ref);
		to get a SecPointer to an instance of your object from the external CFTypeRef.
*/
class SecCFTypes
{
public:
    SecCFTypes();

public:
	/* Add new instances of CFClass here that you want registered with the CF runtime. */
	CFClass Access;
	CFClass ACL;
	CFClass Certificate;
	CFClass Identity;
	CFClass IdentityCursor;
	CFClass ItemImpl;
	CFClass KCCursorImpl;
	CFClass KeychainImpl;
    CFClass PasswordImpl;
	CFClass Policy;
	CFClass PolicyCursor;
	CFClass Trust;
	CFClass TrustedApplication;
	CFClass ExtendedAttribute;
};

extern SecCFTypes &gTypes();

} // end namespace KeychainCore

} // end namespace Security


#endif // !_SECURITY_SECCFTYPES_H_
