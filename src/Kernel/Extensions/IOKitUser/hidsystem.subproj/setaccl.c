/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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
cc -o /tmp/setaccl setaccl.c -Wall -framework IOKit
*/

#include <IOKit/IOKitLib.h>
#include <drivers/event_status_driver.h>
#include <IOKit/hidsystem/IOHIDLib.h>
#include <IOKit/hidsystem/IOHIDParameter.h>
#include <stdio.h>
#include <assert.h>

int
main(int argc, char **argv)
{
    NXEventHandle	hdl;
    double		dbl1;
    CFStringRef		key;

    hdl = NXOpenEventStatus();
    assert( hdl );

    if( argc > 2) {

        if( 't' == argv[2][0])
            key = CFSTR(kIOHIDTrackpadAccelerationType);
        else
            key = CFSTR(kIOHIDMouseAccelerationType);

        assert( KERN_SUCCESS == IOHIDGetAccelerationWithKey(hdl, key, &dbl1));
        printf("IOHIDGetAccelerationWithKey = %f\n", dbl1);
        sscanf( argv[1], "%lf", &dbl1 );
        assert( KERN_SUCCESS == IOHIDSetAccelerationWithKey(hdl, key, dbl1));
        assert( KERN_SUCCESS == IOHIDGetAccelerationWithKey(hdl, key, &dbl1));
        printf("now IOHIDGetAccelerationWithKey = %f\n", dbl1);
    } else {
        assert( KERN_SUCCESS == IOHIDGetMouseAcceleration(hdl, &dbl1));
        printf("IOHIDGetMouseAcceleration = %f\n", dbl1);
        sscanf( argv[1], "%lf", &dbl1 );
        assert( KERN_SUCCESS == IOHIDSetMouseAcceleration(hdl, dbl1));
        assert( KERN_SUCCESS == IOHIDGetMouseAcceleration(hdl, &dbl1));
        printf("now IOHIDGetMouseAcceleration = %f\n", dbl1);
    }

    return( 0 );
}

