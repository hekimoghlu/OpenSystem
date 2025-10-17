/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 6, 2024.
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
#include <sys/cdefs.h>

#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/thread_switch.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <mach/mach_time.h>
#include <syslog.h>
#include <asl.h>
#include <msgtracer_keys.h>

#include <CoreFoundation/CoreFoundation.h>

#include <IOKit/IOKitLib.h>
#include <libkern/OSByteOrder.h>
#include <IOKit/IOMessage.h>
#include <IOKit/IOCFURLAccess.h>
#include <IOKit/graphics/IOGraphicsLib.h>
#include <IOKit/graphics/IOGraphicsLibPrivate.h>


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

kern_return_t
IOFramebufferOpen(
        io_service_t    service,
        task_port_t     owningTask,
        unsigned int    type,
        io_connect_t  * connect )
{
    kern_return_t       kr;

    kr = IOServiceOpen( service, owningTask, type, connect );

    return( kr );
}

void IOLoadPEFsFromURL( CFURLRef ndrvDirURL __unused, io_service_t service __unused )
{
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <IOKit/i2c/IOI2CInterfacePrivate.h>

struct IOI2CConnect
{
    io_connect_t connect;
};


IOReturn IOI2CCopyInterfaceForID( CFTypeRef identifier, io_service_t * interface )
{
    CFMutableDictionaryRef dict, matching;
    mach_port_t            masterPort;
    kern_return_t          kr;
    io_iterator_t          iter;

    IOMasterPort( MACH_PORT_NULL, &masterPort );
    
    matching = IOServiceMatching(kIOI2CInterfaceClassName);
    if(!matching)
        return( kIOReturnNoMemory );

    dict = CFDictionaryCreateMutable( kCFAllocatorDefault, 0,
                &kCFTypeDictionaryKeyCallBacks,
                &kCFTypeDictionaryValueCallBacks);
    if(!dict)
        return( kIOReturnNoMemory );

    CFDictionarySetValue(dict, CFSTR(kIOI2CInterfaceIDKey), identifier);
    CFDictionarySetValue(matching, CFSTR(kIOPropertyMatchKey), dict);
    CFRelease(dict);

    kr = IOServiceGetMatchingServices( masterPort, matching, &iter);
    if( kIOReturnSuccess == kr) {
        *interface = IOIteratorNext( iter );
        IOObjectRelease( iter );
    }

    return( kr );
}

IOReturn IOFBGetI2CInterfaceCount( io_service_t framebuffer, IOItemCount * count )
{
    CFArrayRef   array;

    array = IORegistryEntryCreateCFProperty( framebuffer, CFSTR(kIOFBI2CInterfaceIDsKey),
                                             kCFAllocatorDefault, kNilOptions );
    if( array) {
        *count = CFArrayGetCount(array);
        CFRelease( array );
    } else
        *count = 0;

    return( kIOReturnSuccess );
}

IOReturn IOFBCopyI2CInterfaceForBus( io_service_t framebuffer, IOOptionBits bus, io_service_t * interface )
{
    IOReturn     kr = kIOReturnNoDevice;
    CFArrayRef   array;
    CFIndex      index;
    CFTypeRef    ident;

    array = IORegistryEntryCreateCFProperty( framebuffer, CFSTR(kIOFBI2CInterfaceIDsKey),
                                             kCFAllocatorDefault, kNilOptions );
    if( !array)
        return( kIOReturnNoDevice );

    index = bus & kIOI2CBusNumberMask;

    do {
        if( index >= CFArrayGetCount(array)) {
            kr = kIOReturnNoDevice;
            continue;
        }

        ident = CFArrayGetValueAtIndex(array, index);
        kr = IOI2CCopyInterfaceForID( ident, interface );

    } while( false );

    CFRelease( array );

    return( kr );
}

IOReturn IOI2CInterfaceOpen( io_service_t interface, IOOptionBits options,
                             IOI2CConnectRef * connect )
{
    kern_return_t         kr;
    struct IOI2CConnect * connectRef;

    if( !IOObjectConformsTo(interface, kIOI2CInterfaceClassName))
        return( kIOReturnBadArgument );

    connectRef = calloc(1, sizeof(struct IOI2CConnect));
    if( !connectRef)
        return( kIOReturnNoMemory );

    kr = IOServiceOpen( interface, mach_task_self(), options, &connectRef->connect );

    if( (kr != kIOReturnSuccess) && connectRef) {
        free(connectRef);
        connectRef = NULL;
    }
    *connect = connectRef;

    return( kr );
}

IOReturn IOI2CInterfaceClose( IOI2CConnectRef connect, IOOptionBits options __unused )
{
    kern_return_t kr;

    kr = IOServiceClose( connect->connect );

    free( connect );

    return( kr );
}

IOReturn IOI2CSendRequest( IOI2CConnectRef connect, IOOptionBits options __unused, 
                           IOI2CRequest * request )
{
    kern_return_t kr;
    IOI2CBuffer   buffer;

    if( request->sendBytes > sizeof(buffer.inlineBuffer))
        return( kIOReturnOverrun );
    if( request->replyBytes > sizeof(buffer.inlineBuffer))
        return( kIOReturnOverrun );

    kr = IOConnectCallMethod(connect->connect, 0,       // Index
                NULL,    0, NULL,    0,                 // Input
                NULL, NULL, NULL, NULL);                // Output
    if( kIOReturnSuccess != kr)
        return( kr );

    buffer.request = *request;
    buffer.request.replyBuffer = 0;
    buffer.request.sendBuffer  = 0;

    if( request->sendBytes)
        bcopy( (void *) request->sendBuffer, &buffer.inlineBuffer[0], request->sendBytes );

    size_t len = sizeof( buffer);
    kr = IOConnectCallMethod(connect->connect, 2,       // Index
                NULL,    0, &buffer, len,               // Input
                NULL, NULL, &buffer, &len);             // Output

    if( buffer.request.replyBytes)
        bcopy( &buffer.inlineBuffer[0], (void *)  request->replyBuffer, buffer.request.replyBytes );
    *request = buffer.request;

    return IOConnectCallMethod(connect->connect, 1,     // Index
                NULL,    0, NULL,    0,                 // Input
                NULL, NULL, NULL, NULL);                // Output
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

