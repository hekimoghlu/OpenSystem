/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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
/*
cc -o test HIDParamTest.c -lIOKit
*/

#include <IOKit/IOKitLib.h>
#include <IOKit/hidsystem/IOHIDShared.h>
#include <assert.h>

mach_port_t	masterPort;

io_connect_t OpenEventDriver( void )
{
	register kern_return_t	kr;
	mach_port_t		ev, service, iter;

	assert( KERN_SUCCESS == (
	kr = IOServiceGetMatchingServices( masterPort,
			 IOServiceMatching( kIOHIDSystemClass ), &iter)
	));

	assert(
	service = IOIteratorNext( iter )
	);

	assert( KERN_SUCCESS == (
        kr = IOServiceOpen( service,
			mach_task_self(),
			kIOHIDParamConnectType,
			&ev)
	));

	IOObjectRelease( service );
	IOObjectRelease( iter );

	return( ev );
}


void TestParams( io_connect_t ev )
{
	kern_return_t	kr;
	NXEventData	event;
	IOGPoint       	loc;
	char *		s = "hello ";
	char		c;

	loc.x = 200;
	loc.y = 200;

	assert( KERN_SUCCESS == (
	kr = IOHIDSetMouseLocation( ev, 200, 200 )
	));

	while( (c = *(s++))) {
            event.key.repeat = FALSE;
            event.key.keyCode = 0;
            event.key.charSet = NX_ASCIISET;
            event.key.charCode = c;
            event.key.origCharSet = event.key.charSet;
            event.key.origCharCode = event.key.charCode;

            assert( KERN_SUCCESS == (
            kr = IOHIDPostEvent ( ev, NX_KEYDOWN, loc, &event,
				  FALSE, 0, FALSE )
            ));
            assert( KERN_SUCCESS == (
            kr = IOHIDPostEvent ( ev, NX_KEYUP, loc, &event,
				  FALSE, 0, FALSE )
            ));
	}
}

int
main(int argc, char **argv)
{
	kern_return_t		kr;

	assert( KERN_SUCCESS == (
	kr = IOMasterPort(   bootstrap_port,
			     &masterPort)
	));
	TestParams( OpenEventDriver());

	return( 0 );
}
