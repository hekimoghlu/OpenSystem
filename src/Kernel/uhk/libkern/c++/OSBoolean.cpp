/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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
/* OSBoolean.cpp created by rsulack on Tue Oct 12 1999 */

#include <libkern/c++/OSBoolean.h>
#include <libkern/c++/OSString.h>
#include <libkern/c++/OSSerialize.h>
#include <libkern/c++/OSLib.h>

#define super OSObject

OSDefineMetaClassAndStructorsWithInit(OSBoolean, OSObject, OSBoolean::initialize())
OSMetaClassDefineReservedUnused(OSBoolean, 0);
OSMetaClassDefineReservedUnused(OSBoolean, 1);
OSMetaClassDefineReservedUnused(OSBoolean, 2);
OSMetaClassDefineReservedUnused(OSBoolean, 3);
OSMetaClassDefineReservedUnused(OSBoolean, 4);
OSMetaClassDefineReservedUnused(OSBoolean, 5);
OSMetaClassDefineReservedUnused(OSBoolean, 6);
OSMetaClassDefineReservedUnused(OSBoolean, 7);

static OSBoolean * gOSBooleanTrue  = NULL;
static OSBoolean * gOSBooleanFalse = NULL;

OSBoolean * const & kOSBooleanTrue  = gOSBooleanTrue;
OSBoolean * const & kOSBooleanFalse = gOSBooleanFalse;

void
OSBoolean::initialize()
{
	gOSBooleanTrue = new OSBoolean;
	assert(gOSBooleanTrue);

	if (!gOSBooleanTrue->init()) {
		gOSBooleanTrue->OSObject::free();
		assert(false);
	}
	;
	gOSBooleanTrue->value = true;

	gOSBooleanFalse = new OSBoolean;
	assert(gOSBooleanFalse);

	if (!gOSBooleanFalse->init()) {
		gOSBooleanFalse->OSObject::free();
		assert(false);
	}
	;
	gOSBooleanFalse->value = false;
}

void
OSBoolean::free()
{
	/*
	 * An OSBoolean should never have free() called on it, since it is a shared
	 * object, with two non-mutable instances: kOSBooleanTrue, kOSBooleanFalse.
	 * There will be cases where an incorrect number of releases will cause the
	 * free() method to be called, however, which we must catch and ignore here.
	 */
	assert(false);
}

void
OSBoolean::taggedRetain(__unused const void *tag) const
{
}
void
OSBoolean::taggedRelease(__unused const void *tag, __unused const int when) const
{
}

OSBoolean *
OSBoolean::withBoolean(bool inValue)
{
	return (inValue) ? kOSBooleanTrue : kOSBooleanFalse;
}

bool
OSBoolean::isTrue() const
{
	return value;
}
bool
OSBoolean::isFalse() const
{
	return !value;
}
bool
OSBoolean::getValue() const
{
	return value;
}

bool
OSBoolean::isEqualTo(const OSBoolean *boolean) const
{
	return boolean == this;
}

bool
OSBoolean::isEqualTo(const OSMetaClassBase *obj) const
{
	OSBoolean * boolean;
	if ((boolean = OSDynamicCast(OSBoolean, obj))) {
		return isEqualTo(boolean);
	} else {
		return false;
	}
}

bool
OSBoolean::serialize(OSSerialize *s) const
{
	if (s->binary) {
		return s->binarySerialize(this);
	}

	return s->addString(value ? "<true/>" : "<false/>");
}
