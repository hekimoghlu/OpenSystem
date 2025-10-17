/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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
#include <libkern/OSBase.h>

__BEGIN_DECLS
#include <mach/mach_types.h>
#include <mach/vm_types.h>
#include <mach/kmod.h>

kmod_start_func_t test1_start;
kmod_stop_func_t test1_stop;
__END_DECLS

#include <libkern/c++/OSContainers.h>
#include <iokit/IOLib.h>

const char *testBuffer = ""
    "{ string	= \"this is a 'string' with spaces\";"
    "  string2	= 'this is also a \"string\" with spaces';"
    "  offset	= 16384:32;"
    "  true          = .true.;"
    "  false         = .false.;"
    "  data		= <0123 4567 89abcdef>;"
    "  array		= (1:8, 2:16, 3:32, 4:64 );"
    "  set		= [ one, two, three, four ];"
    "  emptydict	= { }@1;"
    "  emptyarray	= ( )@2;"
    "  emptyset	= [ ]@3;"
    "  emptydata	= < >@4;"
    "  emptydict2	= @1;"
    "  emptyarray2	= @2;"
    "  emptyset2	= @3;"
    "  emptydata2	= @4;"
    "  dict2		= { string = asdfasdf; };"
    "  dict3		= { string = asdfasdf; };"
    "}@0";

kern_return_t
test1_start(struct kmod_info *ki, void *data)
{
	IOLog("test buffer start:\n%s\n:test buffer end.\n", testBuffer);

	// test unserialize
	OSString *errmsg;
	OSObject *d = OSUnserialize(testBuffer, &errmsg);
	if (!d) {
		IOLog("%s\n", errmsg->getCStringNoCopy());
		return KMOD_RETURN_SUCCESS;
	}

	// test serialize
	OSSerialize *s = OSSerialize::withCapacity(5);
	if (!d->serialize(s)) {
		IOLog("serialization failed\n");
		return KMOD_RETURN_SUCCESS;
	}

	IOLog("serialized object's length = %d, capacity = %d\n", s->getLength(), s->getCapacity());
	IOLog("object unformatted = %s\n", s->text());

	// try second time
	OSObject *d2 = OSUnserializeXML(s->text(), &errmsg);
	if (!d2) {
		IOLog("%s\n", errmsg->getCStringNoCopy());
		return KMOD_RETURN_SUCCESS;
	}

	IOLog("\nserialized objects compared %ssuccessfully objectwise\n\n",
	    d->isEqualTo(d2) ? "":"un");

	if (d2) {
		d2->release();
	}
	s->release();
	if (d) {
		d->release();
	}

	return KMOD_RETURN_SUCCESS;
}

kern_return_t
test1_stop(struct kmod_info *ki, void *data)
{
	return KMOD_RETURN_SUCCESS;
}
