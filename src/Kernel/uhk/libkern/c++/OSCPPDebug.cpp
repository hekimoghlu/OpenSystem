/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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
#include <libkern/c++/OSCPPDebug.cpp>

__BEGIN_DECLS

void
OSPrintMemory( void )
{
	OSMetaClass::printInstanceCounts();

	IOLog("\n"
	    "ivar kalloc()       0x%08x\n"
	    "malloc()            0x%08x\n"
	    "containers kalloc() 0x%08x\n"
	    "IOMalloc()          0x%08x\n"
	    "----------------------------------------\n",
	    debug_ivars_size,
	    debug_malloc_size,
	    debug_container_malloc_size,
	    debug_iomalloc_size
	    );
}

__END_DECLS
