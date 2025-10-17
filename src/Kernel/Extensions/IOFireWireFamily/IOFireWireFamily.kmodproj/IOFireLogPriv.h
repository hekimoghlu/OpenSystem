/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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
#ifndef _IOKIT_IOFIRELOGPRIV_H
#define _IOKIT_IOFIRELOGPRIV_H

#if FIRELOGCORE

#include <IOKit/firewire/IOFireLog.h>

#include <libkern/c++/OSObject.h>
#include <IOKit/system.h>

#import <IOKit/firewire/IOFireWireController.h>
#import <IOKit/firewire/IOLocalConfigDirectory.h>

#include <IOKit/IOBufferMemoryDescriptor.h>

#define kFireLogSize (12*1024*1024)    // 8MB
//#define kFireLogSize (512*1024)    // 512KB

class IOFireLog : public OSObject
{
    OSDeclareAbstractStructors(IOFireLog)

protected:

    typedef struct
    {
        UInt32	start;
        UInt32	end;
    } FireLogHeader;
    
    static OSObject * 	sFireLog;
    static int			sTempBufferIndex;
    static char 		sTempBuffer[255];
    
    IOFireWireController *		fController;
    IOBufferMemoryDescriptor * 	fLogDescriptor;
    IOFWAddressSpace * 			fLogPhysicalAddressSpace;
    IOPhysicalAddress			fLogPhysicalAddress;
    FireLogHeader *				fLogBuffer;
    char *						fLogStart;
    char *						fLogEnd;
    IOLocalConfigDirectory *	fUnitDir;
    IOLock *					fLock;
    bool						fNeedSpace;
    UInt32 						fLogSize;
    UInt32						fRandomID;
	
    static void firelog_putc( char c );
    virtual IOReturn initialize( void );

    inline char * logicalToPhysical( char * logical )
        { return (logical - ((char*)fLogBuffer) + ((char*)fLogPhysicalAddress)); }
    inline char * physicalToLogical( char * physical )
        { return (physical - ((char*)fLogPhysicalAddress) + ((char*)fLogBuffer)); }
   
    inline char * encodedToLogical( UInt32 encoded )
        { return ( fLogStart + ((encoded % (kFireLogSize>>2))<<2) ); }

    inline UInt32 sizeToEncoded( UInt32 size )
        { return (size >> 2); }
        
public:

    static IOReturn create( );
    
    virtual void free( void );
    
    static IOFireLog * getFireLog( void );
    
    virtual void setMainController( IOFireWireController * controller );
    virtual IOFireWireController * getMainController( void );
    virtual IOPhysicalAddress getLogPhysicalAddress( void );
    virtual UInt32 getLogSize( void );
    virtual UInt32 getRandomID( void );
	
    virtual void logString( const char *format, va_list ap );
    
};

// IOFireLogPublisher
//
// publishes firelog on a controller
//

class IOFireLogPublisher : public OSObject
{
    OSDeclareAbstractStructors(IOFireLogPublisher)

protected:
    
    IOFireWireController *		fController;
    IOFireLog *					fFireLog;
    IOMemoryDescriptor * 		fLogDescriptor;
    IOFWAddressSpace * 			fLogPhysicalAddressSpace;
    IOLocalConfigDirectory *	fUnitDir;

    virtual IOReturn initWithController( IOFireWireController* controller );
        
public:

    static IOFireLogPublisher * create( IOFireWireController * controller );
    virtual void free( void );
    
};

#endif

#endif
