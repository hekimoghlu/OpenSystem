/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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
cc -g -o /tmp/evstest EVSTests.c -framework IOKit
*/

#include <drivers/event_status_driver.h>
#include <mach/thread_switch.h>
#include <assert.h>

int
main(int argc, char **argv)
{
	NXEventHandle		hdl;
        NXEventSystemDevice	info[ 20 ];
	unsigned int		size;
	double			dbl1, dbl2;
	int			i, j;
	NXKeyMapping		mapping;
	int			mapSize;
	char *			map;
	unsigned int *		pmap;
	NXMouseScaling		scaling;

	hdl = NXOpenEventStatus();
	assert( hdl );

	size = sizeof( info) / sizeof( int);
	assert( NXEventSystemInfo(hdl, NX_EVS_DEVICE_INFO, info, &size ));
	size = size * sizeof( int) / sizeof( info[0]);
	printf("%d devices\n", size);
	for( i = 0; i < size; i++) {
		printf("%d : dev_type = %d, interface = %d, "
			"id = %d, interface_addr = %d\n", i,
		info[ i ].dev_type, info[ i ].interface,
		info[ i ].id, info[ i ].interface_addr );
	}

	dbl1 = NXKeyRepeatInterval(hdl);
	printf("NXKeyRepeatInterval = %f\n", dbl1);
	NXSetKeyRepeatInterval( hdl, 1.0 / 4 );
	dbl2 = NXKeyRepeatInterval(hdl);
	printf("now NXKeyRepeatInterval = %f\n", dbl2);

	dbl1 = NXKeyRepeatThreshold(hdl);
	printf("NXKeyRepeatThreshold = %f\n", dbl1);
	NXSetKeyRepeatThreshold( hdl, 1.0 );
	dbl2 = NXKeyRepeatThreshold(hdl);
	printf("now NXKeyRepeatThreshold = %f\n", dbl2);

	assert( KERN_SUCCESS == IOHIDGetMouseAcceleration(hdl, &dbl1));
	printf("IOHIDGetMouseAcceleration = %f\n", dbl1);

	assert( KERN_SUCCESS == IOHIDSetMouseAcceleration(hdl, 1.0));
	assert( KERN_SUCCESS == IOHIDGetMouseAcceleration(hdl, &dbl1));
	printf("now IOHIDGetMouseAcceleration = %f\n", dbl1);

	NXGetMouseScaling(hdl, &scaling);
	printf("Scaling[ %d ]: ", scaling.numScaleLevels);
	for( i = 0; i < scaling.numScaleLevels; i++)
	    printf("(%d,%d), ",
		scaling.scaleThresholds[i], scaling.scaleFactors[i]);
	printf("\n");
	assert( KERN_SUCCESS == IOHIDSetMouseAcceleration(hdl, 0.3));
	NXSetMouseScaling(hdl, &scaling);
	printf("Scaling[ %d ]: ", scaling.numScaleLevels);
	for( i = 0; i < scaling.numScaleLevels; i++)
	    printf("(%d,%d), ",
		scaling.scaleThresholds[i], scaling.scaleFactors[i]);
	printf("\n");
	assert( KERN_SUCCESS == IOHIDGetMouseAcceleration(hdl, &dbl1));
	printf("now IOHIDGetMouseAcceleration = %f\n", dbl1);


	printf("NXKeyRepeatThreshold = %f\n", dbl1);

	printf("NXAutoDimThreshold = %f\n", NXAutoDimThreshold(hdl));
	printf("NXAutoDimTime = %f\n", NXAutoDimTime(hdl));
	printf("NXIdleTime = %f\n", NXIdleTime(hdl));
	printf("NXAutoDimState = %d\n", NXAutoDimState(hdl));
	printf("NXAutoDimBrightness = %f\n", NXAutoDimBrightness(hdl));
	printf("NXScreenBrightness = %f\n", NXScreenBrightness(hdl));

	NXSetAutoDimThreshold( hdl, 200.0 );
//        NXSetAutoDimState( hdl, 1 );
        NXSetAutoDimBrightness( hdl, 0.5 );
        NXSetScreenBrightness( hdl, 0.7 );

	printf("now NXAutoDimThreshold = %f\n", NXAutoDimThreshold(hdl));
	printf("NXAutoDimTime = %f\n", NXAutoDimTime(hdl));
	printf("NXIdleTime = %f\n", NXIdleTime(hdl));
	printf("NXAutoDimState = %d\n", NXAutoDimState(hdl));
	printf("NXAutoDimBrightness = %f\n", NXAutoDimBrightness(hdl));
	printf("NXScreenBrightness = %f\n", NXScreenBrightness(hdl));

	mapSize = NXKeyMappingLength(hdl);
	map = (char *) malloc( mapSize );
	mapping.mapping = map;
	mapping.size = mapSize;
	assert( &mapping == NXGetKeyMapping(hdl, &mapping));

	pmap = (unsigned int *) map;
if(0)	while ((((char *)pmap) - map) < mapSize) {
	    printf("%04x: ", ((char *)pmap) - map);
	    for( j = 0; j < 8; j++ ) {
		printf("%08x ", *pmap++);
		if( (((char *)pmap) - map) >= mapSize)
		    break;
	    }
	    printf("\n");
	}

	map[ 0x32 ] = 0x62;	// a == b
	assert( &mapping == NXSetKeyMapping(hdl, &mapping));

	printf("sleeping...\n");
	thread_switch( 0, SWITCH_OPTION_WAIT, 10 * 1000 );

	NXResetKeyboard(hdl);
	NXResetMouse(hdl);

	return( 0 );
}
